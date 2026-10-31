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

export type EmotionReading = {
  text: string;
  emotion: Emotion;
  confidence: number;
  scores: EmotionScores;
};

export type RawReading = {
  index: number;
  emotion: string;
  scores: Partial<EmotionScores>;
};

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

  return { text, emotion: best, confidence: scores[best], scores };
}

export function topEmotionDistribution(readings: EmotionReading[]): Record<Emotion, number> {
  const counts = Object.fromEntries(EMOTIONS.map((e) => [e, 0])) as Record<Emotion, number>;
  for (const reading of readings) counts[reading.emotion] += 1;
  return counts;
}
