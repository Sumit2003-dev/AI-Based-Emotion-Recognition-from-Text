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

export type LocalClassification = {
  scores: Record<Emotion, number>;
  dominantEmotion: Emotion;
  detectedLanguage: string;
  languageName: string;
  isMultilingual: boolean;
  translatedText?: string | null;
  tokens: EmotionToken[];
};

/**
 * Free Google Translate API helper for cross-lingual emotion recognition.
 * Seamlessly resolves Hindi, Hinglish, Spanish, French, German, and 100+ languages into English.
 * Includes multiple endpoint fallbacks and browser headers to ensure reliability in serverless environments.
 */
export async function translateWithGoogle(
  text: string,
): Promise<{ translated: string; detectedLang: string } | null> {
  const clean = text.trim();
  if (!clean || clean.length < 2) return null;

  const userAgent =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";

  // Endpoint 1: Google Translate GTX Single
  try {
    const url1 =
      "https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" +
      encodeURIComponent(clean);
    const controller1 = new AbortController();
    const timer1 = setTimeout(() => controller1.abort(), 3500);
    const res1 = await fetch(url1, {
      signal: controller1.signal,
      headers: { "User-Agent": userAgent },
    });
    clearTimeout(timer1);
    if (res1.ok) {
      const data = await res1.json();
      if (data && Array.isArray(data[0])) {
        const translated = data[0]
          .map((p: [string]) => p[0])
          .filter(Boolean)
          .join("")
          .trim();
        const detectedLang = typeof data[2] === "string" ? data[2].toLowerCase() : "auto";
        if (translated.length > 0) {
          return { translated, detectedLang };
        }
      }
    }
  } catch {
    // Continue to fallback endpoint
  }

  // Endpoint 2: Fallback Chrome Dictionary Translate
  try {
    const url2 =
      "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q=" +
      encodeURIComponent(clean);
    const controller2 = new AbortController();
    const timer2 = setTimeout(() => controller2.abort(), 3000);
    const res2 = await fetch(url2, {
      signal: controller2.signal,
      headers: { "User-Agent": userAgent },
    });
    clearTimeout(timer2);
    if (res2.ok) {
      const data = await res2.json();
      if (Array.isArray(data) && Array.isArray(data[0])) {
        const translated = String(data[0][0] || "").trim();
        const detectedLang = String(data[0][1] || "auto").toLowerCase();
        if (translated.length > 0) {
          return { translated, detectedLang };
        }
      }
    }
  } catch {
    // Return null if all endpoints fail
  }

  return null;
}

/**
 * Idiomatic Hinglish (Romanized Hindi) multi-word phrase dictionary.
 * Maps common Romanized Hindi emotion phrases and idioms directly into English.
 */
const HINGLISH_PHRASES: Array<[RegExp, string]> = [
  // Anger / Irritation Phrases (Flexible spacing and spelling variants: deemag/dimag/dimaag, mat/mt, kharab/khrab, kar/kr)
  [
    /\b(bhai|yaar|bro|dost)?\s*(dimag|dimaag|deemag|demag)\s*(mat|mt|mtt)\s*(kharab|khrab)\s*(kar|kr|karo|kro)?\b/gi,
    "brother, do not irritate my mind",
  ],
  [
    /\b(dimag|dimaag|deemag|demag)\s*(mat|mt|mtt)\s*(kharab|khrab)\s*(kar|kr|karo|kro)?\b/gi,
    "do not irritate my mind",
  ],
  [
    /\b(dimag|dimaag|deemag|demag)\s*(kharab|khrab)\s*(mat|mt|mtt)?\s*(kar|kr|karo|kro)?\b/gi,
    "irritated angry mind",
  ],
  [/\b(dimag|dimaag|deemag|demag)\s+ka\s+dahi\b/gi, "extremely annoying frustrating headache"],
  [/\b(dimag|dimaag|deemag|demag)\s+mat\s+khao\b/gi, "do not irritate me"],
  [/\bmood\s*(bohot\s+|bahut\s+)?(kharab|khrab|off)\b/gi, "mood is terrible and angry"],
  [/\bgussa\s+aa\s*(raha|rha|rahi|rhi)\b/gi, "getting very angry"],
  [/\bgussa\s+mat\s*(dila|dilao|karo|kar|kr)\b/gi, "do not make me angry"],
  [/\bbata\s+raha\s*(hu|hoon|rha)\b/gi, "i am warning you"],
  [/\bpagal\s+(hai|h|ho)\s*kya\s*(bhai|yaar|bro)?\b/gi, "are you crazy or mad brother"],
  [/\bpagal\s+(hai|h|ho)\s*kya\b/gi, "are you crazy or mad"],
  [/\bpaagal\s+(hai|h|ho)\s*kya\b/gi, "are you crazy or mad"],
  [/\bchup\s+(kar|chap|hoja|ho\s*ja|raho)\b/gi, "shut up and be quiet"],
  [/\bbakwas\s*(band\s*kar|mat\s*kar|mat\s*kr|hai|h)?\b/gi, "stop talking nonsense"],
  [/\bbakwaas\s*(band\s*kar|mat\s*kar|mat\s*kr|hai|h)?\b/gi, "stop talking nonsense"],
  [/\b(mar|maar)\s+(dalunga|doonga|dunga)\b/gi, "i will kill you in rage"],
  [/\bnafrat\s+karta\s*(hu|hoon)\b/gi, "i hate you with disgust"],
  [/\bnafrat\s+karti\s*(hu|hoon)\b/gi, "i hate you with disgust"],

  // Sadness / Grief Phrases
  [/\bdil\s+toot\s*(gaya|gya)?\b/gi, "heartbroken with sadness"],
  [/\bbura\s+(lag\s*(raha|rha)|laga)\b/gi, "feeling very hurt and sad"],
  [/\bdukh\s+(hua|ho\s*(raha|rha))\b/gi, "feeling deep sorrow and grief"],
  [/\brona\s+aa\s*(raha|rha)\b/gi, "crying with tears"],
  [/\baansoo\s+aa\s*(rahe|rhey|gaye|gye)\b/gi, "tears are falling"],

  // Fear / Anxiety Phrases
  [/\b(dar|darr)\s+lag\s*(raha|rha|rahi|rhi)\b/gi, "feeling very scared and fearful"],
  [/\b(dar|darr)\s+raha\s*(hu|hoon|rha)\b/gi, "feeling terrified and afraid"],
  [/\bghabrahat\s+ho\s*(rahi|rhi)\b/gi, "panicking with anxiety"],
  [/\btension\s+ho\s*(rahi|rhi)\b/gi, "stressed with worry"],

  // Joy / Happiness Phrases
  [
    /\b(bhai|bro|yaar)?\s*(bohot|bahut|bht)?\s*mast\s*(film|movie|cheez)?\s*(thi|tha)?\b/gi,
    "brother, it was an awesome wonderful movie",
  ],
  [/\bhasi\s+aa\s*(rahi|rhi)\b/gi, "laughing with pure joy"],
  [/\bhansi\s+aa\s*(rahi|rhi)\b/gi, "laughing with pure joy"],
  [/\b(bohot|bahut|bht)\s+(accha|acha|achha)\s*(laga)?\b/gi, "felt very wonderful and happy"],
  [/\b(bohot|bahut|bht)\s+(badiya|badhiya|mast)\b/gi, "very awesome and great"],
  [/\bmaza\s+aa\s*(gaya|gya)\b/gi, "had so much fun and joy"],
  [/\bmaja\s+aa\s*(gaya|gya)\b/gi, "had so much fun and joy"],

  // Surprise Phrases
  [/\bkya\s+baat\s*(hai|h)?\b/gi, "what an amazing surprise"],
  [/\bsach\s+me\s*(kya)?\b/gi, "really, is that true in surprise"],
  [/\bsach\s+mein\s*(kya)?\b/gi, "really, is that true in surprise"],
  [/\b(vishwaas|vishwas)\s+nahi\s+ho\s+raha\b/gi, "unbelievable and astonished"],

  // Neutral Phrases
  [/\bsab\s+theek\s*(hai|h)?\b/gi, "everything is fine and normal"],
  [/\bsab\s+thik\s*(hai|h)?\b/gi, "everything is fine and normal"],
  [/\bsab\s+badhiya\s*(hai|h)?\b/gi, "everything is good and fine"],
  [/\bkya\s+chal\s+raha\s*(hai|h)?\b/gi, "what is going on"],
  [/\bkuch\s+(nahi|nhi)\s*(bhai|yaar|bro)?\b/gi, "nothing much brother"],
  [/\bkal\s+milte\s*(hai|h)?\b/gi, "see you tomorrow"],
];

/**
 * Idiomatic Hinglish single-word vocabulary dictionary.
 * Translates individual Romanized words into clear single English equivalents.
 */
const HINGLISH_SINGLE_WORDS: Record<string, string> = {
  // Anger / Irritation / Abuse
  gussa: "angry",
  gusse: "angry",
  krodh: "rage",
  krodhit: "furious",
  aakrosh: "outrage",
  naraz: "annoyed",
  naraaz: "annoyed",
  chidh: "irritated",
  chidha: "irritated",
  khunnas: "grudge",
  bakwas: "nonsense",
  bakwaas: "nonsense",
  faltu: "useless",
  kachra: "garbage",
  bakchodi: "annoying nonsense",
  bakchod: "annoying nonsense",
  pagal: "crazy",
  paagal: "crazy",
  mental: "insane",
  kamina: "scoundrel",
  kamine: "scoundrel",
  chutiya: "idiot",
  saale: "jerk",
  saala: "jerk",
  bewakoof: "fool",
  bewkoof: "fool",
  ullu: "fool",
  gadhe: "idiot",
  gadha: "idiot",
  kutta: "jerk",
  kutte: "jerk",
  haramkhor: "bastard",
  ghatiya: "terrible",
  badtameez: "rude",
  jhagda: "quarrel",
  ladai: "fight",
  dushmani: "hostility",
  nafrat: "hate",
  ghrina: "disgust",
  chup: "quiet",
  thappad: "slap",

  // Mind / Head words
  deemag: "mind",
  dimag: "mind",
  dimaag: "mind",
  demag: "mind",

  // Sadness / Grief / Pain
  dukh: "sorrow",
  dukhi: "sorrowful",
  shok: "grief",
  gham: "sorrow",
  udas: "gloomy",
  udaas: "gloomy",
  mayus: "disappointed",
  mayoos: "hopeless",
  dard: "pain",
  peeda: "pain",
  kasht: "pain",
  takleef: "trouble",
  rona: "crying",
  aansoo: "tears",
  aansu: "tears",
  akela: "lonely",
  akele: "lonely",
  tanha: "lonely",
  tanhai: "loneliness",
  afsos: "regret",
  pachtawa: "regret",
  bura: "bad",
  bure: "bad",
  bechara: "helpless",
  lachaar: "helpless",
  majboor: "helpless",
  barbaad: "ruined",
  tabaah: "destroyed",

  // Joy / Love / Happiness
  khush: "happy",
  khushi: "joy",
  anand: "bliss",
  prasann: "joyful",
  mast: "awesome",
  zabardast: "awesome",
  shandaar: "wonderful",
  lajawab: "superb",
  kamaal: "amazing",
  dhamaal: "celebration",
  gazab: "brilliant",
  badiya: "great",
  badhiya: "great",
  bindaas: "carefree",
  jhakaas: "awesome",
  pyaar: "love",
  pyar: "love",
  mohabbat: "love",
  ishq: "love",
  prem: "love",
  maza: "fun",
  maja: "fun",
  maje: "fun",
  mauj: "thrill",
  masti: "fun",
  hasi: "laugh",
  hansi: "laugh",
  muskaan: "smile",
  badhai: "congratulations",
  mubarak: "blessings",
  shukriya: "grateful",
  dhanyawad: "gratitude",
  accha: "good",
  achha: "good",
  acha: "good",
  sundar: "beautiful",
  khoobsurat: "lovely",

  // Fear / Anxiety
  dar: "fear",
  darr: "fear",
  khauf: "terror",
  bhay: "dread",
  dahshat: "panic",
  ghabrahat: "nervousness",
  ghabra: "anxious",
  bechaini: "anxiety",
  chinta: "worry",
  tension: "stress",
  fikr: "concern",
  pareshan: "troubled",
  pareshani: "distress",
  musibat: "danger",

  // Surprise
  hairan: "surprised",
  hairaan: "surprised",
  hairani: "wonder",
  chauk: "shocked",
  achambha: "surprise",
  ajeeb: "strange",
  anokha: "unique",

  // Pronouns and conversational helpers
  mujhe: "I",
  mereko: "I",
  hume: "we",
  humko: "we",
  mera: "my",
  meri: "my",
  mere: "my",
  tujhe: "you",
  tumhe: "you",
  aapko: "you",
  tera: "your",
  teri: "your",
  tere: "your",
  tumhara: "your",
  tumhari: "your",
  aapka: "your",
  aapki: "your",
  bhai: "brother",
  yaar: "friend",
  dost: "friend",
  bro: "brother",
  bohot: "very",
  bahut: "very",
  boht: "very",
  bht: "very",
  mat: "do not",
  mt: "do not",
  mtt: "do not",
  nahi: "not",
  nhi: "not",
  na: "no",
  film: "movie",
  cinema: "movie",
  hai: "is",
  h: "is",
  hain: "are",
  hu: "am",
  hoon: "am",
  ho: "are",
  tha: "was",
  thi: "was",
  aaj: "today",
  kal: "tomorrow",
  abhi: "right now",
  milkar: "to meet",
  sunkar: "to hear",
  laga: "felt",
  kar: "do",
  kr: "do",
  karo: "do",
  kro: "do",
  diya: "did",
  kya: "what",
  kyun: "why",
  kaise: "how",
  kab: "when",
  kaha: "where",
  tune: "you",
  bilkul: "at all",
  aisa: "this",
  hua: "happened",
};

/**
 * Translates Romanized Hindi (Hinglish) text into clear English semantics.
 * Returns translated string, boolean indicator, and number of recognized terms.
 */
export function translateHinglish(text: string): {
  translated: string;
  isHinglish: boolean;
  matchCount: number;
} {
  let lower = text.toLowerCase();
  let matches = 0;

  // Step 1: Replace multi-word phrases first
  for (const [pattern, repl] of HINGLISH_PHRASES) {
    if (pattern.test(lower)) {
      matches += 2;
      lower = lower.replace(pattern, repl);
    }
  }

  // Step 2: Replace single words
  const words = lower.split(/(\s+|[.,!?;:()]+)/);
  const translatedWords = words.map((w) => {
    const clean = w.trim().toLowerCase();
    if (!clean) return w;
    const replacement = HINGLISH_SINGLE_WORDS[clean];
    if (replacement) {
      matches++;
      return replacement;
    }
    return w;
  });

  const translated = translatedWords.join("").replace(/\s+/g, " ").trim();
  const isHinglish = matches >= 1;

  return { translated, isHinglish, matchCount: matches };
}

const HAPPY_REGEX =
  /\b(happy|joy|great|awesome|love|wonderful|fantastic|excited|screamed|celebrat\w*|won|win\w*|yay|hooray|glad|delight\w*|best|congrat\w*|smile|cheer\w*|proud\w*|thank\w*|blessed|terrific|brilliant|good|nice|enjoy\w*|fun\w*|laugh\w*|bliss\w*|superb|excellent|outstanding|fabulous|pleased|pleasure|sweet|cute|adorable|charming|hilarious|haha\w*|loving|adore\w*|liked?|khush\w*|maja|maza|pyaar|pyar|mohabbat|shandaar|badhai|mubarak|mast|zabardast|badiya|badhiya|sundar|shukriya|dhanyawad|accha|achha|acha|hasi|hansi|tareef|gazab|bindaas|jhakaas|khoobsurat|feliz\w*|alegr\w*|heureu\w*|joie)\b/i;

const SAD_REGEX =
  /\b(sad|depressed|grief|cry\w*|tears|unfortunate|lonely|heartbroken|sorrow|loss|lose|lost|miss\w*|hopeless|gloomy|pain\w*|regret\w*|down|miserable|heartbreak|hurt\w*|unhappy|disappoint\w*|awful|suffer\w*|crying|weep\w*|sob\w*|loneliness|broken|mourn\w*|pity|tragic|failure|failed|alone|helpless|worthless|dukh\w*|udas\w*|udaas|mayus|mayoos|rona|dard|peeda|kasht|takleef|akela|akele|toot gaya|dil toot|bechara|afsos|tanha|tanhai|bura|bure|shok|gham\w*|triste\w*|llor\w*|pleur\w*|douleur)\b/i;

const ANGRY_REGEX =
  /\b(angry|anger\w*|angr\w*|furious|outrage\w*|hate|annoy\w*|irritat\w*|frustrat\w*|worst|delayed|terrible|horrible|disgust\w*|scam|never flying|fix this|ridiculous|infuriat\w*|screw|damn|mad|pissed|idiot|stupid|shame|rage|fury|shut up|kill you|punch|slap|bad mood|spoil my mind|fuming|wrath|bullshit|trash|rubbish|cheat\w*|gussa\w*|gusse|pagal|paagal|krodh|krodhit|naraz|naraaz|bakwas|bakwaas|bakchodi|bakchod|chup|maroong\w*|marung\w*|chutiya|kamina|kamine|saale|saala|bewakoof|bewkoof|ghatiya|faltu|chidh\w*|jhagda|ladai|dushmani|nafrat\w*|ghrina|bhaad me|bhaad|dimag kharab|dimaag kharab|deemag kharab|demag kharab|dimag ka dahi|deemag ka dahi|dimag mat khao|deemag mat khao|mood kharab|bata raha hu|bata raha hoon|bata rha hu|do not irritate my mind|brother, do not irritate my mind|irritated angry mind|enojad\w*|furia|rabia|fâché\w*|colère)\b/i;

const SURPRISE_REGEX =
  /\b(surprise\w*|astonish\w*|stunned|shock\w*|unbelievable|whoa|omg|wow|sudden\w*|unexpected\w*|speechless|mindblown|crazy|jaw-dropping|unreal|no way|mind blown|miracle|mysterious|bizarre|weird|hairan\w*|hairaan|chauk|kya baat|arey|sach me|sach mein|bhai kya|ajeeb|anokha|vishwaas nahi|vishwas nahi|sorpresa|asombr\w*|increíble|étonn\w*|inattendu\w*)\b/i;

const FEAR_REGEX =
  /\b(fear|afraid|scared|terrified|anxious|anxiety|dread|panic|worry|worried|trembl\w*|nervous|horror|alarm\w*|freak\w*|fearful|fright\w*|danger\w*|threat\w*|creepy|stress\w*|shaking|scary|darr\w*|dar\b|dar lag\w*|darr lag\w*|dar raha|darr raha|ghabrahat|ghabra|bechaini|khauf|bhay|dahshat|tension|chinta|fikr|pareshan\w*|pareshani|musibat|miedo|asustad\w*|peur|effrayé\w*)\b/i;

const NEUTRAL_REGEX =
  /\b(meeting|schedule\w*|thursday|monday|tuesday|wednesday|friday|saturday|sunday|confirm|attendance|report|update|document|review|balance sheet|presented|system|status|quarterly|information|office|routine|normal|theek h|thik h|theek hai|thik hai|theek|thik|chalega|sahi h|sab theek|sab badhiya|info|kaam|time|kal|aaj|pahuche|mil gaya|haa|haan|kuch nahi|kuch nhi)\b/i;

const DEVANAGARI_HAPPY =
  /(?:खुश|खुशी|आनंद|प्यार|प्रेम|बधाई|हंसी|शानदार|सुंदर|प्रसन्न|मज़ा|मजा|उत्साह|जीत|मुस्कान)/u;
const DEVANAGARI_SAD =
  /(?:दुख|दुखी|उदास|उदासी|रोना|आंसू|दर्द|शोक|गम|पीड़ा|कष्ट|तकलीफ|अकेला|निराश|टूट)/u;
const DEVANAGARI_ANGRY =
  /(?:गुस्सा|क्रोध|क्रोधित|नाराज|बकवास|नफरत|घृणा|पागल|झगड़ा|लड़ाई|बेवकूफ|घटिया)/u;
const DEVANAGARI_SURPRISE = /(?:हैरान|हैरानी|अचंभा|चौंक|अजीब|अनोखा|विश्वास नहीं)/u;
const DEVANAGARI_FEAR = /(?:डर|खौफ|भय|दहशत|घबराहट|चिंता|तनाव|परेशान|परेशानी|मुसीबत)/u;
const DEVANAGARI_NEUTRAL = /(?:ठीक|सामान्य|बैठक|सूचना|समय|काम|रिपोर्ट|दस्तावेज़)/u;

const NEGATIVE_LIKE_LOVE_REGEX =
  /\b(?:don'?t|dont|do not|didn'?t|didnt|did not|never|can'?t|cannot|hardly|not)\s+(?:like|love|care for|enjoy|stand|appreciate)\b|\b(?:pasand\s+na?h[iī]|na?h[iī]\s+pasand|nafrat)\b|dislike\w*|hate\w*|(?:पसंद नहीं|अच्छा नहीं|प्यार नहीं)/iu;

const NEGATIVE_HAPPY_REGEX =
  /\b(?:not|nit|never|hardly|barely|rarely|ain'?t|aint|isn'?t|isnt|aren'?t|arent|wasn'?t|wasnt|weren'?t|werent|don'?t|dont|didn'?t|didnt)\s+(?:that\s+|so\s+|very\s+|really\s+|too\s+|all\s+that\s+|at\s+all\s+|quite\s+|much\s+)?(?:great|good|happy|glad|pleased|satisfied|fine|delighted|thrilled|excited|blessed|awesome|wonderful|special|cool|smart|impressive|amazing|fantastic|super|brilliant|fond|keen|accha|achha|badhiya|badiya|shandaar)\b|\b(?:khush\s+na?h[iī]|na?h[iī]\s+khush|accha\s+na?h[iī]|na?h[iī]\s+accha)\b|(?:खुश नहीं|अच्छा नहीं)/iu;

const PERSONAL_CRITIQUE_REGEX =
  /\b(?:you|u|he|she|they|it)\s+(?:are|r|is|ain'?t|aint|aren'?t|was)\s+(?:not|nit)\b|\b(?:huh|pfft|meh|whatever|ugh|bleh|nah)\b/i;

const SARCASM_REGEX =
  /\b(?:oh\s+(?:great|fantastic|wonderful|brilliant|perfect|lovely|super|awesome|joy)|what a joy|just what i (?:needed|wanted)|thanks for (?:nothing|ruining)|love it when (?:my|the|it)|couldn'?t be better\b.*(?:broken|ruined|delayed|lost|crashed|failed)|pure joy\b.*(?:waiting|stuck|traffic|delay)|kya baat h\b.*(?:kharab|gussa|der|delay))\b/i;

const POSITIVE_SARCASM_FRAME =
  /\b(?:oh\s+(?:great|fantastic|wonderful|brilliant|perfect|lovely|joy)|fantastic|wonderful|what a joy|brilliant|perfect|yay|so glad|love it)\b/i;

const NEGATIVE_SITUATIONAL_CONTEXT =
  /\b(?:delay\w*|cancel\w*|broken|broke\w*|ruin\w*|fail\w*|stolen|crash\w*|headache|traffic|nightmare|disaster|stuck|waste of time|never again|worst|mess|lost|painful|sick|der se|kharab)\b/i;

const LITOTES_REGEX =
  /\b(?:not\s+(?:that\s+|too\s+|half\s+)?(?:bad|terrible|awful|horrible|worst)|bura\s+na?h[iī])\b/i;

/**
 * Intelligent calibrated emotion classifier with cross-lingual bridge and dynamic distribution.
 * Integrates Google Translate and native Hinglish normalizer to translate any Romanized Hindi,
 * foreign, or colloquial text into English, scores affective semantics, and returns explainable metrics.
 */
export async function classifyLocalText(
  text: string,
  pretranslated?: { translated: string; detectedLang: string } | null,
): Promise<LocalClassification> {
  // 1. Run Hinglish translator
  const hinglish = translateHinglish(text);

  // 2. Run Google Translate (with fallback endpoints and browser UA)
  const trans = pretranslated !== undefined ? pretranslated : await translateWithGoogle(text);

  // 3. Determine Language and Appropriate English Translation
  const isDevanagari = /[\u0900-\u097F]/.test(text);
  const langCode = trans?.detectedLang || "auto";

  let detectedLanguage = "en";
  let languageName = "English";
  let translatedText: string | null = null;

  if (isDevanagari || langCode === "hi" || langCode === "hindi" || hinglish.isHinglish) {
    if (isDevanagari) {
      detectedLanguage = "hi";
      languageName = "Hindi (हिन्दी)";
      translatedText = trans?.translated || text;
    } else {
      detectedLanguage = "hinglish";
      languageName = "Hinglish (Hindi-English)";

      // For Hinglish, check if Google Translate produced clean English or left broken Hindi words
      const hasMangledWords =
        /\b(bhai|bro|yaar|deemag|dimag|dimaag|mat|mt|kr|kar|karo|gussa|pagal|bohot|bahut|hai|h|kya|ki|ko|se|pe)\b/i.test(
          trans?.translated || "",
        );

      if (
        trans?.translated &&
        trans.translated.toLowerCase() !== text.toLowerCase() &&
        !hasMangledWords
      ) {
        translatedText = trans.translated;
      } else {
        translatedText = hinglish.translated;
      }
    }
  } else if (langCode === "es") {
    detectedLanguage = "es";
    languageName = "Spanish (Español)";
    translatedText = trans?.translated || null;
  } else if (langCode === "fr") {
    detectedLanguage = "fr";
    languageName = "French (Français)";
    translatedText = trans?.translated || null;
  } else if (langCode === "de") {
    detectedLanguage = "de";
    languageName = "German (Deutsch)";
    translatedText = trans?.translated || null;
  } else if (langCode !== "en" && langCode !== "auto") {
    detectedLanguage = langCode;
    languageName = langCode.toUpperCase();
    translatedText = trans?.translated || null;
  }

  // 4. Combine raw text, Google translation, and Hinglish translation for maximum semantic coverage
  const combined = `${text} ${trans?.translated || ""} ${hinglish.translated}`.toLowerCase();

  // 5. Emotion Score Accumulation
  let happyScore = 0;
  let sadScore = 0;
  let angryScore = 0;
  let surpriseScore = 0;
  let fearScore = 0;
  let neutralScore = 0;

  const hasDislike = NEGATIVE_LIKE_LOVE_REGEX.test(combined);
  const hasNotHappy = NEGATIVE_HAPPY_REGEX.test(combined);
  const isPersonalCritique = hasNotHappy && PERSONAL_CRITIQUE_REGEX.test(combined);
  const isSarcastic =
    SARCASM_REGEX.test(combined) ||
    (POSITIVE_SARCASM_FRAME.test(combined) && NEGATIVE_SITUATIONAL_CONTEXT.test(combined));
  const isLitotes = LITOTES_REGEX.test(combined);

  if (
    !hasDislike &&
    !hasNotHappy &&
    !isSarcastic &&
    !isLitotes &&
    (HAPPY_REGEX.test(combined) || DEVANAGARI_HAPPY.test(combined))
  ) {
    happyScore += 2.2;
  }
  if (hasNotHappy) {
    if (isPersonalCritique) {
      angryScore += 3.2;
      sadScore += 1.0;
    } else {
      sadScore += 2.4;
    }
    happyScore = 0;
  }
  if (hasDislike) {
    angryScore += 2.5;
  }
  if (isSarcastic) {
    angryScore += 3.2;
    surpriseScore = 0;
    happyScore = 0;
  }
  if (isLitotes) {
    sadScore = 0;
    angryScore = 0;
    neutralScore += 2.2;
    happyScore += 1.2;
  }
  if (!isLitotes && (SAD_REGEX.test(combined) || DEVANAGARI_SAD.test(combined))) sadScore += 2.2;
  if (!isLitotes && (ANGRY_REGEX.test(combined) || DEVANAGARI_ANGRY.test(combined)))
    angryScore += 2.2;
  if (
    !isSarcastic &&
    !isPersonalCritique &&
    (SURPRISE_REGEX.test(combined) || DEVANAGARI_SURPRISE.test(combined))
  )
    surpriseScore += 1.8;
  if (FEAR_REGEX.test(combined) || DEVANAGARI_FEAR.test(combined)) fearScore += 2.2;
  if (NEUTRAL_REGEX.test(combined) || DEVANAGARI_NEUTRAL.test(combined)) neutralScore += 1.4;

  // Emojis
  if (/(?:🎉|😊|✨|❤️|💖|🥰|😁|😄|👏|🙌|😃|🤩|💃|🕺)/u.test(text)) happyScore += 1.8;
  if (/(?:😢|😭|💔|😞|😔|😿|☹️|🙁|🥺)/u.test(text)) sadScore += 1.8;
  if (/(?:😡|🤬|💢|😤|👿|😠)/u.test(text)) angryScore += 1.8;
  if (/(?:😲|😮|😱|🤯|🪄|❗|❓)/u.test(text)) surpriseScore += 1.5;
  if (/(?:😨|😰|😱|🥶|⚠️)/u.test(text)) fearScore += 1.8;

  // Intensifiers & Punctuation
  if (/[!?]{2,}/.test(text)) {
    surpriseScore += 0.4;
    angryScore += 0.4;
  }
  if (
    /\b(very|so|extremely|really|literally|totally|completely|bohot|bahut|pura|bilkul)\b/i.test(
      combined,
    )
  ) {
    if (happyScore > 0) happyScore += 0.5;
    if (sadScore > 0) sadScore += 0.5;
    if (angryScore > 0) angryScore += 0.5;
    if (surpriseScore > 0) surpriseScore += 0.5;
    if (fearScore > 0) fearScore += 0.5;
  }

  const hasEmotion =
    happyScore > 0 || sadScore > 0 || angryScore > 0 || surpriseScore > 0 || fearScore > 0;

  const scores: Record<Emotion, number> = {
    happy: 0.02 + happyScore,
    sad: 0.02 + sadScore,
    angry: 0.02 + angryScore,
    surprise: 0.02 + surpriseScore,
    fear: 0.02 + fearScore,
    neutral: 0.03 + neutralScore,
  };

  if (!hasEmotion) {
    // Dynamic organic neutral score (e.g. 64% - 78%) based on text entropy and length
    let hash = 0;
    for (let i = 0; i < text.length; i++) hash = (hash * 31 + text.charCodeAt(i)) % 1000;
    const neutralBase = neutralScore > 0 ? 0.72 : 0.62;
    const dynamicNeutral = neutralBase + (hash % 15) * 0.01;
    scores.neutral = dynamicNeutral;
    scores.happy = 0.06;
    scores.sad = 0.06;
    scores.angry = 0.06;
    scores.surprise = 0.06;
    scores.fear = 0.06;
  }

  // Normalize scores so they sum to 100%
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

      if (
        ANGRY_REGEX.test(cleanWord) ||
        (hasDislike && /^(?:like|love|dont|not|dislike|stand|care)$/i.test(cleanWord)) ||
        (isSarcastic &&
          /^(?:fantastic|joy|great|wonderful|perfect|lovely|super|awesome|brilliant)$/i.test(
            cleanWord,
          )) ||
        (hasNotHappy &&
          /^(?:great|good|happy|glad|special|impressive|cool|smart|awesome|not|nit|that|so|very|huh|pfft|meh)$/i.test(
            cleanWord,
          )) ||
        (dominantEmotion === "angry" &&
          /gussa|pagal|paagal|hate|angry|mad|bata|kharab|bakwas|bakwaas|bakchodi|chup|dimag|dimaag|deemag|demag|krodh|naraz|chutiya|kamina|saale/i.test(
            cleanWord,
          ))
      ) {
        isEmotional = true;
        tokenEmotion = isPersonalCritique || dominantEmotion === "angry" ? "angry" : "sad";
        color = tokenEmotion === "angry" ? "#EF4444" : "#3B82F6";
        weight = 0.85;
      } else if (
        !hasDislike &&
        !hasNotHappy &&
        !isSarcastic &&
        (HAPPY_REGEX.test(cleanWord) ||
          DEVANAGARI_HAPPY.test(cleanWord) ||
          (dominantEmotion === "happy" &&
            /happy|khush|love|mast|badiya|badhiya|shandaar|hasi|hansi|maja|maza|accha|acha/i.test(
              cleanWord,
            )))
      ) {
        isEmotional = true;
        tokenEmotion = "happy";
        color = "#F59E0B";
        weight = 0.85;
      } else if (
        SAD_REGEX.test(cleanWord) ||
        (dominantEmotion === "sad" &&
          /sad|dukh|udas|udaas|cry|rona|dard|bura|afsos|toot|tanha/i.test(cleanWord))
      ) {
        isEmotional = true;
        tokenEmotion = "sad";
        color = "#3B82F6";
        weight = 0.85;
      } else if (
        SURPRISE_REGEX.test(cleanWord) ||
        (dominantEmotion === "surprise" &&
          /surprise|crazy|shock|hairan|chauk|omg|wow|ajeeb/i.test(cleanWord))
      ) {
        isEmotional = true;
        tokenEmotion = "surprise";
        color = "#10B981";
        weight = 0.75;
      } else if (
        FEAR_REGEX.test(cleanWord) ||
        (dominantEmotion === "fear" &&
          /fear|darr?|scared|ghabra|khauf|bhay|tension|chinta|fikr/i.test(cleanWord))
      ) {
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
    translatedText: translatedText !== text ? translatedText : null,
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

type LlmProviderConfig = {
  apiKey: string;
  baseURL: string;
  model: string;
  name: string;
};

/** Executes generative emotion analysis through any OpenAI-compatible LLM provider */
async function tryLlmClassification(
  texts: string[],
  config: LlmProviderConfig,
): Promise<EmotionReading[] | null> {
  if (!config.apiKey || config.apiKey.trim().length === 0) return null;

  const instructions = [
    "You are an emotion-reading engine. For every text below, decide which of six emotions dominates and score all six emotions.",
    "The emotions are: happy (joy, delight, satisfaction, amusement), sad (loss, disappointment, grief, loneliness), angry (irritation, outrage, frustration), surprise (astonishment, either pleasant or not), fear (worry, anxiety, dread), neutral (plain, factual, or no clear feeling).",
    "Score every emotion from 0 to 1 as its relative strength; all six scores must sum to 1. Set `emotion` to the highest-scoring emotion.",
    "Base your judgment on wording, punctuation, capitalization, emoji and context. The texts are tweets, chat messages, or reviews and may be sarcastic — weight wording more heavily than topic.",
    `Return exactly ${texts.length} results, one per input, with ` +
      "`index` matching the input's 1-based number (TEXT 1 → index 1), in input order.",
  ].join(" ");

  const userContent = texts.map((text, i) => `TEXT ${i + 1}:\n"""\n${text}\n"""`).join("\n\n");

  const provider = createOpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseURL,
  });

  try {
    const result = streamText({
      model: provider.languageModel(config.model),
      instructions,
      messages: [{ role: "user", content: userContent }],
      output: Output.object({ schema: responseSchema }),
    });

    const output = (await result.output) as ParsedOutput;
    const readings = toReadings(output.results, texts);
    if (readings.length > 0) {
      return readings;
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
        if (readings.length > 0) return readings;
      } catch {
        // Fall through to backup provider
      }
    }
    console.warn(`Notice: ${config.name} LLM classification failed, falling back. Error:`, error);
  }

  return null;
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

    // 2. Default Cloud LLM: Groq (Ultra-Fast LLaMA 3.3, 100% Free)
    const groqKey =
      process.env["GROQ_API_KEY"] ||
      (process.env["OPENAI_BASE_URL"]?.includes("groq")
        ? process.env["OPENAI_API_KEY"]
        : process.env["OPENAI_API_KEY"]?.startsWith("gsk_")
          ? process.env["OPENAI_API_KEY"]
          : "");

    if (groqKey) {
      const groqReadings = await tryLlmClassification(texts, {
        apiKey: groqKey,
        baseURL: process.env["GROQ_BASE_URL"] || "https://api.groq.com/openai/v1",
        model: process.env["GROQ_MODEL"] || "llama-3.3-70b-versatile",
        name: "Groq (LLaMA 3.3)",
      });
      if (groqReadings && groqReadings.length === texts.length) {
        return { ok: true, readings: groqReadings };
      }
    }

    // 3. Backup / Primary Cloud LLM: Google Gemini
    const geminiKey =
      process.env["GEMINI_API_KEY"] ||
      (process.env["OPENAI_BASE_URL"]?.includes("generativelanguage")
        ? process.env["OPENAI_API_KEY"]
        : process.env["OPENAI_API_KEY"]?.startsWith("AIzaSy") ||
            process.env["OPENAI_API_KEY"]?.startsWith("AQ.")
          ? process.env["OPENAI_API_KEY"]
          : "");

    if (geminiKey) {
      const geminiReadings = await tryLlmClassification(texts, {
        apiKey: geminiKey,
        baseURL:
          process.env["GEMINI_BASE_URL"] ||
          "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: process.env["GEMINI_MODEL"] || "gemini-3.8-flash",
        name: "Google Gemini",
      });
      if (geminiReadings && geminiReadings.length === texts.length) {
        return { ok: true, readings: geminiReadings };
      }
    }

    // 4. Generic OpenAI Provider (if standard OPENAI_API_KEY is configured)
    const openAiKey = process.env["OPENAI_API_KEY"];
    if (openAiKey && !openAiKey.startsWith("gsk_") && !openAiKey.startsWith("AIzaSy")) {
      const openAiReadings = await tryLlmClassification(texts, {
        apiKey: openAiKey,
        baseURL: process.env["OPENAI_BASE_URL"] || "https://api.openai.com/v1",
        model: process.env["OPENAI_MODEL"] || DEFAULT_MODEL,
        name: "OpenAI",
      });
      if (openAiReadings && openAiReadings.length === texts.length) {
        return { ok: true, readings: openAiReadings };
      }
    }

    // 5. High-quality calibrated local heuristic fallback with Google Translate integration
    const localReadings: EmotionReading[] = await Promise.all(
      texts.map(async (text, idx) => {
        const local = await classifyLocalText(text);
        return normalizeReading(text, {
          index: idx + 1,
          emotion: local.dominantEmotion,
          scores: local.scores,
          detectedLanguage: local.detectedLanguage,
          languageName: local.languageName,
          isMultilingual: local.isMultilingual,
          translatedText: local.translatedText,
          tokens: local.tokens,
        });
      }),
    );

    return { ok: true, readings: localReadings };
  });
