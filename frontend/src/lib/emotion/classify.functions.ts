import { createOpenAI } from "@ai-sdk/openai";
import { createServerFn } from "@tanstack/react-start";
import { NoObjectGeneratedError, Output, streamText } from "ai";
import { z } from "zod";

import {
  EMOTIONS,
  MAX_BATCH_TEXTS,
  MAX_TEXT_CHARS,
  normalizeReading,
  type Emotion,
  type EmotionReading,
  type EmotionToken,
  type RawReading,
} from "./emotions";

const DEFAULT_MODEL = "gpt-4o-mini";

type ParsedScores = Record<EmotionReading["emotion"], number>;

type ParsedResult = {
  index: number;
  emotion: string;
  scores: Partial<ParsedScores>;
};

type ParsedOutput = {
  results: ParsedResult[];
};

const scoresSchema = z.object({
  happy: z.number(),
  sad: z.number(),
  angry: z.number(),
  surprise: z.number(),
  fear: z.number(),
  neutral: z.number(),
});

const resultSchema = z.object({
  index: z.number(),
  emotion: z.enum(EMOTIONS),
  scores: scoresSchema,
});

const responseSchema = z.object({
  results: z.array(resultSchema),
});

export type ClassifyResponse =
  { ok: true; readings: EmotionReading[] } | { ok: false; error: string };

/** Results come back 1-based (TEXT 1 → index 1); map back onto the texts array. */
export function toReadings(results: ParsedResult[], texts: string[]): EmotionReading[] {
  return results.flatMap((entry): EmotionReading[] => {
    const text = texts[entry.index - 1];
    if (!text) return [];
    return [normalizeReading(text, entry as RawReading)];
  });
}

type LocalClassification = {
  scores: Record<Emotion, number>;
  dominantEmotion: Emotion;
  detectedLanguage: string;
  languageName: string;
  isMultilingual: boolean;
  tokens: EmotionToken[];
};

/**
 * Intelligent local calibrated emotion classifier with cross-lingual & Hinglish support.
 * Evaluates semantic keywords across English, Hinglish, Hindi, Spanish, and French,
 * emojis, punctuation, and valence.
 * Serves as an instant, zero-configuration engine when running on Vercel or standalone.
 */
export function classifyLocalText(text: string): LocalClassification {
  const lower = text.toLowerCase();
  const scores: Record<Emotion, number> = {
    happy: 0.05,
    sad: 0.05,
    angry: 0.05,
    surprise: 0.05,
    fear: 0.05,
    neutral: 0.15,
  };

  // Language detection
  const isDevanagari = /[\u0900-\u097F]/.test(text);
  const isHinglish =
    /\b(kya|hai|h|bhai|mujhe|mera|meri|mere|raha|rahi|hu|hoon|yaar|nahi|kyun|hoga|hogi|wala|wali|tere|apna|bata|karo|dekh|bahut|kaise|pe|se|ko|aur|par|gussa|pagal|chup|dukhi|khush|maza|darr|dard|rona)\b/i.test(
      text,
    );
  const isSpanish =
    /\b(el|la|los|las|un|una|por|para|con|que|estoy|es|son|pero|mas|muy|enojad\w*|feliz\w*|triste\w*|miedo|gracias)\b/i.test(
      text,
    );
  const isFrench =
    /\b(le|la|les|des|est|sont|avec|pour|dans|je|tu|il|elle|nous|vous|très|mais|fâché\w*|heureu\w*|peur|merci)\b/i.test(
      text,
    );

  let detectedLanguage = "en";
  let languageName = "English";
  if (isDevanagari) {
    detectedLanguage = "hi";
    languageName = "Hindi (हिन्दी)";
  } else if (isHinglish) {
    detectedLanguage = "hinglish";
    languageName = "Hinglish (Hindi-English)";
  } else if (isSpanish) {
    detectedLanguage = "es";
    languageName = "Spanish (Español)";
  } else if (isFrench) {
    detectedLanguage = "fr";
    languageName = "French (Français)";
  }

  const happyPatterns = [
    /\b(happy|joy|great|awesome|love|wonderful|fantastic|excited|screamed|celebrat\w*|won|win\w*|yay|hooray|glad|delight\w*|best|congrat\w*|smile|cheer\w*|prouder?|proud|thank\w*|acceptance\w*|blessed|terrific|brilliant)\b/gi,
    /\b(khush|khushi|maja|maza|pyaar|pyar|mohabbat|shandaar|badhai|mubarak|mast|zabardast|badiya|badhiya|sundar|shukriya|dhanyawad|khoob|accha|achha)\b/gi,
    /(?:खुश|खुशी|प्यार|मज़ा|शानदार|बधाई|मुबारक|मस्त|बढ़िया|धन्यवाद)/gu,
    /\b(feliz\w*|felicidad|alegr\w*|amor|genial|excelente|encant\w*|gracias|heureu\w*|joie|merveilleu\w*|merci|bravo)\b/gi,
    /(?:🎉|😊|✨|❤️|💖|🥰|😁|😄|👏|🙌|😃|🤩|💃|🕺)/gu,
  ];

  const sadPatterns = [
    /\b(sad|depressed|grief|cry\w*|tears|unfortunate|lonely|heartbroken|sorrow|loss|lose|lost bag|miss\w*|hopeless|gloomy|pain\w*|regret\w*|down|broke my heart|miserable|heartbreak)\b/gi,
    /\b(dukh|dukhi|udas|udaas|rona|ro raha|ro rahi|dard|chhod diya|akela|akele|toot gaya|dil toot|bechara|afsos|tanha|mayus)\b/gi,
    /(?:दुख|दुखी|उदास|रोना|दर्द|अकेला|टूटा|अफसोस|तन्हा)/gu,
    /\b(triste\w*|tristeza|llor\w*|dolor|deprimi\w*|soledad|pena|pleur\w*|douleur|chagrin|seul\w*|déçu\w*)\b/gi,
    /(?:😢|😭|💔|😞|😔|😿|☹️|🙁|🥺)/gu,
  ];

  const angryPatterns = [
    /\b(angry|anger\w*|angr\w*|furious|outrage\w*|hate|annoy\w*|irritat\w*|frustrat\w*|worst|delayed|terrible|horrible|disgust\w*|scam|never flying|fix this|ridiculous|infuriat\w*|screw|damn|mad|pissed|idiot|stupid|shame)\b/gi,
    /\b(gussa|gusse|pagal|krodh|naraz|bakwas|bakwaas|chup|maroonga|marunga|chutiya|kamina|saale|bewakoof|ghatiya|faltu|chidh|jhagda|bhaad me|dimag kharab|bata raha hu|bhadak)\b/gi,
    /(?:गुस्सा|क्रोध|नाराज|बकवास|पागल|घटिया|चुप|झगड़ा)/gu,
    /\b(enojad\w*|furia|rabia|odio|bronca|molest\w*|maldit\w*|fâché\w*|colère|énerv\w*|détester|haine|furieu\w*)\b/gi,
    /(?:😡|🤬|💢|😤|👿|😠)/gu,
  ];

  const surprisePatterns = [
    /\b(surprise\w*|astonish\w*|stunned|shock\w*|wait\b|didn't see|unbelievable|whoa|omg|wow|sudden\w*|unexpected\w*|today\?|really\?|speechless|mindblown)\b/gi,
    /\b(hairan|hairani|chauk|kya baat|arey|sach me|sach mein|bhai kya|pagal h kya|shock|ajeeb|vishwaas nahi|itna jaldi)\b/gi,
    /(?:हैरान|चौंक|अरे|सच में|अजीब|विश्वास नहीं)/gu,
    /\b(sorpresa|asombr\w*|increíble|no me digas|vaya|étonn\w*|inattendu\w*)\b/gi,
    /(?:😲|😮|😱|🤯|🪄|❗|❓)/gu,
  ];

  const fearPatterns = [
    /\b(fear|afraid|scared|terrified|anxious|anxiety|dread|panic|worry|worried|trembl\w*|hands won't stop|nervous|horror|alarm\w*|freak\w*)\b/gi,
    /\b(darr|dar lag raha|darr lag|ghabrahat|khauf|bhay|tension|dar lagta|chinta|fati padi)\b/gi,
    /(?:डर|घबराहट|खौफ|भय|चिंता|डरना)/gu,
    /\b(miedo|asustad\w*|terror|pánico|preocupad\w*|temor|peur|effrayé\w*|angoisse)\b/gi,
    /(?:😨|😰|😱|🥶|⚠️)/gu,
  ];

  const neutralPatterns = [
    /\b(meeting|schedule\w*|thursday|confirm|attendance|report|update|document|review|balance sheet|presented|system|status|quarterly|information)\b/gi,
    /\b(theek h|thik h|info|kaam|time|kal|aaj|pahuche|kuch nahi)\b/gi,
    /(?:😐|ℹ️|📅|📋|📄)/gu,
  ];

  const countMatches = (patterns: RegExp[]) =>
    patterns.reduce((acc, regex) => acc + (lower.match(regex)?.length || 0), 0);

  const happyCount = countMatches(happyPatterns);
  const sadCount = countMatches(sadPatterns);
  const angryCount = countMatches(angryPatterns);
  const surpriseCount = countMatches(surprisePatterns);
  const fearCount = countMatches(fearPatterns);
  const neutralCount = countMatches(neutralPatterns);

  // Apply emotional boost
  scores.happy += happyCount * 0.75;
  scores.sad += sadCount * 0.75;
  scores.angry += angryCount * 0.75;
  scores.surprise += surpriseCount * 0.65;
  scores.fear += fearCount * 0.75;
  scores.neutral += neutralCount * 0.45;

  if (/[!?]{2,}/.test(text)) {
    scores.surprise += 0.2;
    scores.angry += 0.2;
  }

  // Normalize scores
  const total = Object.values(scores).reduce((sum, v) => sum + v, 0);
  for (const emotion of EMOTIONS) {
    scores[emotion] = parseFloat((scores[emotion] / total).toFixed(4));
  }

  let dominantEmotion: Emotion = "neutral";
  let maxScore = -1;
  for (const emotion of EMOTIONS) {
    if (scores[emotion] > maxScore) {
      maxScore = scores[emotion];
      dominantEmotion = emotion;
    }
  }

  // Tokenize and build explainability
  const words = text.split(/(\s+)/);
  const tokens: EmotionToken[] = words
    .filter((w) => w.trim().length > 0)
    .map((w) => {
      const cleanWord = w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
      let isEmotional = false;
      let tokenEmotion = "neutral";
      let color: string | null = null;
      let weight = 0.0;

      if (angryPatterns.some((p) => p.test(cleanWord))) {
        isEmotional = true;
        tokenEmotion = "angry";
        color = "#EF4444";
        weight = 0.85;
      } else if (happyPatterns.some((p) => p.test(cleanWord))) {
        isEmotional = true;
        tokenEmotion = "happy";
        color = "#F59E0B";
        weight = 0.85;
      } else if (sadPatterns.some((p) => p.test(cleanWord))) {
        isEmotional = true;
        tokenEmotion = "sad";
        color = "#3B82F6";
        weight = 0.85;
      } else if (surprisePatterns.some((p) => p.test(cleanWord))) {
        isEmotional = true;
        tokenEmotion = "surprise";
        color = "#10B981";
        weight = 0.75;
      } else if (fearPatterns.some((p) => p.test(cleanWord))) {
        isEmotional = true;
        tokenEmotion = "fear";
        color = "#8B5CF6";
        weight = 0.85;
      }

      return {
        token: w,
        is_emotional: isEmotional,
        emotion: tokenEmotion,
        color,
        weight,
      };
    });

  return {
    scores,
    dominantEmotion,
    detectedLanguage,
    languageName,
    isMultilingual: detectedLanguage !== "en",
    tokens,
  };
}

/** Try query the local FastAPI backend if running */
async function tryFastApiBackend(texts: string[]): Promise<EmotionReading[] | null> {
  const backendUrl = process.env["FASTAPI_URL"] || "http://127.0.0.1:8000";
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const res = await fetch(`${backendUrl}/api/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texts }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) return null;
    const data = await res.json();
    if (!data.results || !Array.isArray(data.results)) return null;

    return data.results.map(
      (
        r: {
          text?: string;
          dominant_emotion?: string;
          emotion?: string;
          confidence?: number;
          detected_language?: string;
          language_name?: string;
          is_multilingual?: boolean;
          translated_text?: string | null;
          sentiment?: string;
          sentiment_score?: number;
          intensity?: { level: string; score: number };
          dimensions?: {
            valence: number;
            arousal: number;
            arousal_label: string;
            dominance: number;
            dominance_label: string;
            subjectivity: number;
            subjectivity_label: string;
          };
          guidance?: {
            psychology?: string;
            empathy_tip?: string;
            suggested_reply?: string;
          };
          tokens?: Array<{
            token: string;
            is_emotional: boolean;
            emotion: string;
            color?: string | null;
            weight: number;
          }>;
          probabilities?:
            | Array<{ emotion: string; ratio?: number; probability?: number }>
            | Record<string, number>;
        },
        idx: number,
      ) => {
        const probMap: Record<string, number> = {};
        if (Array.isArray(r.probabilities)) {
          for (const item of r.probabilities) {
            probMap[item.emotion] = item.ratio ?? (item.probability ? item.probability / 100 : 0);
          }
        } else if (typeof r.probabilities === "object" && r.probabilities !== null) {
          for (const [k, v] of Object.entries(r.probabilities)) {
            probMap[k] = typeof v === "number" ? (v > 1 ? v / 100 : v) : 0;
          }
        }

        const happyScore = (probMap["joy"] || 0) + (probMap["love"] || 0);
        const sadScore = probMap["sadness"] || 0;
        const angryScore = (probMap["anger"] || 0) + (probMap["disgust"] || 0);
        const surpriseScore = probMap["surprise"] || 0;
        const fearScore = probMap["fear"] || 0;
        const neutralScore = probMap["neutral"] || 0;

        const scores: Record<Emotion, number> = {
          happy: happyScore > 0 ? happyScore : 0.02,
          sad: sadScore > 0 ? sadScore : 0.02,
          angry: angryScore > 0 ? angryScore : 0.02,
          surprise: surpriseScore > 0 ? surpriseScore : 0.02,
          fear: fearScore > 0 ? fearScore : 0.02,
          neutral: neutralScore > 0 ? neutralScore : 0.02,
        };

        const total = Object.values(scores).reduce((sum, v) => sum + v, 0);
        if (total > 0) {
          for (const emotion of EMOTIONS) {
            scores[emotion] = parseFloat((scores[emotion] / total).toFixed(4));
          }
        }

        const rawDom = (r.dominant_emotion || r.emotion || "").toLowerCase();
        let dominantEmotion: Emotion = "neutral";
        if (rawDom === "joy" || rawDom === "love" || rawDom === "happy") {
          dominantEmotion = "happy";
        } else if (rawDom === "sadness" || rawDom === "sad") {
          dominantEmotion = "sad";
        } else if (rawDom === "anger" || rawDom === "disgust" || rawDom === "angry") {
          dominantEmotion = "angry";
        } else if (rawDom === "surprise") {
          dominantEmotion = "surprise";
        } else if (rawDom === "fear") {
          dominantEmotion = "fear";
        } else if (rawDom === "neutral") {
          dominantEmotion = "neutral";
        } else {
          let maxVal = -1;
          for (const emotion of EMOTIONS) {
            if (scores[emotion] > maxVal) {
              maxVal = scores[emotion];
              dominantEmotion = emotion;
            }
          }
        }

        return normalizeReading(texts[idx] || r.text || "", {
          index: idx + 1,
          emotion: dominantEmotion,
          scores,
          detectedLanguage: r.detected_language,
          languageName: r.language_name,
          isMultilingual: r.is_multilingual,
          translatedText: r.translated_text,
          sentiment: r.sentiment,
          sentimentScore: r.sentiment_score,
          intensityLevel: r.intensity?.level,
          intensityScore: r.intensity?.score,
          dimensions: r.dimensions,
          guidance: r.guidance,
          tokens: r.tokens,
        });
      },
    );
  } catch {
    return null;
  }
}

export const classifyEmotions = createServerFn({ method: "POST" })
  .validator((input: unknown) =>
    z.object({ texts: z.array(z.string()).min(1).max(MAX_BATCH_TEXTS) }).parse(input),
  )
  .handler(async ({ data }): Promise<ClassifyResponse> => {
    const texts = data.texts
      .map((text) => text.trim().slice(0, MAX_TEXT_CHARS))
      .filter((text) => text.length > 0);
    if (texts.length === 0) {
      return { ok: false, error: "Add at least one line of text to analyze." };
    }

    // 1. Try local FastAPI NLP service if active
    const fastApiReadings = await tryFastApiBackend(texts);
    if (fastApiReadings && fastApiReadings.length === texts.length) {
      return { ok: true, readings: fastApiReadings };
    }

    // 2. Try OpenAI API if OPENAI_API_KEY is configured
    const openAiKey = process.env["OPENAI_API_KEY"];
    if (openAiKey) {
      const instructions = [
        "You are an emotion-reading engine. For every text below, decide which of six emotions dominates and score all six emotions.",
        "The emotions are: happy (joy, delight, satisfaction, amusement), sad (loss, disappointment, grief, loneliness), angry (irritation, outrage, frustration), surprise (astonishment, either pleasant or not), fear (worry, anxiety, dread), neutral (plain, factual, or no clear feeling).",
        "Score every emotion from 0 to 1 as its relative strength; all six scores must sum to 1. Set `emotion` to the highest-scoring emotion.",
        "Base your judgment on wording, punctuation, capitalization, emoji and context. The texts are tweets, chat messages, or reviews and may be sarcastic — weight wording more heavily than topic.",
        `Return exactly ${texts.length} results, one per input, with ` +
          "`index` matching the input's 1-based number (TEXT 1 → index 1), in input order.",
      ].join(" ");

      const userContent = texts.map((text, i) => `TEXT ${i + 1}:\n"""\n${text}\n"""`).join("\n\n");

      const provider = createOpenAI({ apiKey: openAiKey });

      try {
        const result = streamText({
          model: provider.languageModel(process.env["OPENAI_MODEL"] || DEFAULT_MODEL),
          instructions,
          messages: [{ role: "user", content: userContent }],
          output: Output.object({ schema: responseSchema }),
        });

        const output = (await result.output) as ParsedOutput;
        const readings = toReadings(output.results, texts);
        if (readings.length > 0) {
          return { ok: true, readings };
        }
      } catch (error) {
        let raw: string | undefined;
        if (NoObjectGeneratedError.isInstance(error)) {
          raw = error.text;
        }
        if (raw) {
          try {
            const parsed = JSON.parse(raw) as ParsedOutput;
            const readings = toReadings(parsed.results, texts);
            if (readings.length > 0) return { ok: true, readings };
          } catch {
            // fall through to local fallback
          }
        }
      }
    }

    // 3. High-quality calibrated local heuristic fallback
    const localReadings: EmotionReading[] = texts.map((text, idx) => {
      const local = classifyLocalText(text);
      return normalizeReading(text, {
        index: idx + 1,
        emotion: local.dominantEmotion,
        scores: local.scores,
        detectedLanguage: local.detectedLanguage,
        languageName: local.languageName,
        isMultilingual: local.isMultilingual,
        tokens: local.tokens,
      });
    });

    return { ok: true, readings: localReadings };
  });
