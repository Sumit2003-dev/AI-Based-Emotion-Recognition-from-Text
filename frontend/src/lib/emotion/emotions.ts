export const EMOTIONS = ["happy", "sad", "angry", "surprise", "fear", "neutral"] as const;

export type Emotion = (typeof EMOTIONS)[number];

/** Limits shared by the analyzer UI and the server-side classifier. */
export const MAX_BATCH_TEXTS = 40;
export const MAX_TEXT_CHARS = 1200;

export const EMOTION_META: Record<Emotion, { label: string; blurb: string }> = {
  happy: { label: "Happy", blurb: "Joy, delight, satisfaction or amusement." },
  sad: { label: "Sad", blurb: "Loss, disappointment, grief or loneliness." },
  angry: { label: "Angry", blurb: "Irritation, outrage or frustration." },
  surprise: { label: "Surprise", blurb: "Astonishment — pleasant or not." },
  fear: { label: "Fear", blurb: "Worry, anxiety or dread." },
  neutral: { label: "Neutral", blurb: "Plainly stated, no strong feeling." },
};

export type EmotionScores = Record<Emotion, number>;

export type AffectiveDimensions = {
  valence: number;
  arousal: number;
  arousal_label: string;
  dominance: number;
  dominance_label: string;
  subjectivity: number;
  subjectivity_label: string;
};

export type EmotionalGuidance = {
  psychology?: string | undefined;
  empathy_tip?: string | undefined;
  suggested_reply?: string | undefined;
};

export type EmotionToken = {
  token: string;
  is_emotional: boolean;
  emotion: string;
  color?: string | null | undefined;
  weight: number;
};

export type EmotionReading = {
  text: string;
  emotion: Emotion;
  confidence: number;
  scores: EmotionScores;
  detectedLanguage?: string | undefined;
  languageName?: string | undefined;
  isMultilingual?: boolean | undefined;
  translatedText?: string | null | undefined;
  sentiment?: string | undefined;
  sentimentScore?: number | undefined;
  intensityLevel?: string | undefined;
  intensityScore?: number | undefined;
  dimensions?: AffectiveDimensions | undefined;
  guidance?: EmotionalGuidance | undefined;
  tokens?: EmotionToken[] | undefined;
};

export type RawReading = {
  index: number;
  emotion: string;
  scores: Partial<EmotionScores>;
  detectedLanguage?: string | undefined;
  languageName?: string | undefined;
  isMultilingual?: boolean | undefined;
  translatedText?: string | null | undefined;
  sentiment?: string | undefined;
  sentimentScore?: number | undefined;
  intensityLevel?: string | undefined;
  intensityScore?: number | undefined;
  dimensions?: AffectiveDimensions | undefined;
  guidance?: EmotionalGuidance | undefined;
  tokens?: EmotionToken[] | undefined;
};

/** Compute fallback affective dimensions for an emotion when backend dimensions are absent. */
function computeDefaultDimensions(emotion: Emotion, confidence: number): AffectiveDimensions {
  const vadMap: Record<Emotion, [number, number, string, number, string]> = {
    happy: [0.85, 72, "High Surge", 75, "Assertive / Empowered"],
    sad: [-0.8, 25, "Calm / Low", 20, "Submissive / Helpless"],
    angry: [-0.75, 88, "High Surge", 70, "Assertive / Empowered"],
    surprise: [0.2, 82, "High Surge", 45, "Balanced Agency"],
    fear: [-0.82, 85, "High Surge", 15, "Submissive / Helpless"],
    neutral: [0.0, 15, "Calm / Low", 50, "Balanced Agency"],
  };
  const [val, aro, aroLabel, dom, domLabel] = vadMap[emotion];
  const confScale = Math.min(1, Math.max(0.3, confidence));
  const subjPct = emotion === "neutral" ? 5 : Math.round(confScale * 90);
  return {
    valence: parseFloat((val * confScale).toFixed(3)),
    arousal: Math.round(aro * confScale),
    arousal_label: aroLabel,
    dominance: Math.round(dom * confScale),
    dominance_label: domLabel,
    subjectivity: subjPct,
    subjectivity_label:
      subjPct >= 65 ? "High Emotion" : subjPct >= 35 ? "Balanced" : "Objective Fact",
  };
}

/** Compute fallback empathy guidance when backend guidance is absent. */
function computeDefaultGuidance(emotion: Emotion): EmotionalGuidance {
  const guidanceMap: Record<Emotion, EmotionalGuidance> = {
    happy: {
      psychology: "Reflects satisfaction, gratitude, accomplishment, or shared social bonding.",
      empathy_tip: "Validate and amplify their excitement. Celebrate their milestone genuinely.",
      suggested_reply: "That is fantastic news! So excited for you, congratulations! 🎉",
    },
    sad: {
      psychology:
        "Indicates feelings of grief, isolation, disappointment, or perceived loss of control.",
      empathy_tip: "Offer non-judgmental presence without rushing to fix everything immediately.",
      suggested_reply:
        "I am really sorry to hear you are dealing with this. Sending strength and here if you need anything.",
    },
    angry: {
      psychology:
        "Driven by perceived injustice, broken expectations, blocked goals, or violated boundaries.",
      empathy_tip:
        "De-escalate immediately. Acknowledge their frustration without being defensive.",
      suggested_reply:
        "I completely understand why this is frustrating. Let's make this right immediately.",
    },
    surprise: {
      psychology: "Cognitive schema disruption following an unexpected event or revelation.",
      empathy_tip: "Help anchor the surprise and clarify next steps or consequences.",
      suggested_reply: "Wow, that really came out of nowhere! How are you processing it so far?",
    },
    fear: {
      psychology: "Anticipation of threat, uncertainty, vulnerability, or impending loss.",
      empathy_tip: "Provide grounding assurance, transparency, and safety cues.",
      suggested_reply:
        "Take a deep breath — we will get through this step by step. You are not alone in this.",
    },
    neutral: {
      psychology: "Informational, pragmatic, and objective communication without affective charge.",
      empathy_tip: "Keep the dialogue clear, efficient, and direct.",
      suggested_reply: "Understood, noted. Thank you for the update.",
    },
  };
  return guidanceMap[emotion];
}

/** Clamp scores into 0..1, renormalize to sum 1, and pick the argmax emotion. */
export function normalizeReading(text: string, raw: RawReading): EmotionReading {
  const scores = {} as EmotionScores;
  let total = 0;
  for (const emotion of EMOTIONS) {
    const value = raw.scores[emotion];
    const clamped =
      typeof value === "number" && Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
    scores[emotion] = clamped;
    total += clamped;
  }
  if (total <= 0) {
    scores.neutral = 1;
    total = 1;
  }
  for (const emotion of EMOTIONS) {
    scores[emotion] = scores[emotion] / total;
  }

  let best: Emotion = EMOTIONS.includes(raw.emotion as Emotion)
    ? (raw.emotion as Emotion)
    : "neutral";
  for (const emotion of EMOTIONS) {
    if (scores[emotion] > scores[best]) best = emotion;
  }

  const confidence = scores[best];
  const sentiment =
    raw.sentiment || (best === "happy" ? "Positive" : best === "neutral" ? "Neutral" : "Negative");
  const sentimentScore =
    raw.sentimentScore !== undefined
      ? raw.sentimentScore
      : best === "happy"
        ? 0.75
        : best === "neutral"
          ? 0.0
          : -0.75;
  const intensityScore = raw.intensityScore ?? Math.round(confidence * 100);
  const intensityLevel =
    raw.intensityLevel ||
    (intensityScore >= 75 ? "High / Strong" : intensityScore >= 45 ? "Moderate" : "Subtle / Low");

  return {
    text,
    emotion: best,
    confidence,
    scores,
    detectedLanguage: raw.detectedLanguage,
    languageName: raw.languageName,
    isMultilingual: raw.isMultilingual,
    translatedText: raw.translatedText,
    sentiment,
    sentimentScore,
    intensityLevel,
    intensityScore,
    dimensions: raw.dimensions || computeDefaultDimensions(best, confidence),
    guidance: raw.guidance || computeDefaultGuidance(best),
    tokens: raw.tokens || [],
  };
}

export function topEmotionDistribution(readings: EmotionReading[]): Record<Emotion, number> {
  const counts = Object.fromEntries(EMOTIONS.map((e) => [e, 0])) as Record<Emotion, number>;
  for (const reading of readings) counts[reading.emotion] += 1;
  return counts;
}
