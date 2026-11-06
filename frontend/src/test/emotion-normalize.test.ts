import { describe, expect, it } from "vitest";

import { EMOTIONS, normalizeReading } from "@/lib/emotion/emotions";

describe("normalizeReading", () => {
  it("keeps exactly six emotion scores that sum to 1", () => {
    const reading = normalizeReading("I am over the moon!", {
      index: 0,
      emotion: "happy",
      scores: { happy: 5, sad: 2, angry: 1, surprise: 1, fear: 1, neutral: 0 },
    });
    expect(EMOTIONS).toHaveLength(8);
    expect(Object.keys(reading.scores).sort()).toEqual([...EMOTIONS].sort());
    const total = EMOTIONS.reduce((sum, e) => sum + reading.scores[e], 0);
    expect(total).toBeCloseTo(1, 6);
  });

  it("picks the highest-scoring emotion, not the label the model claimed", () => {
    const reading = normalizeReading("text", {
      index: 0,
      emotion: "angry",
      scores: { happy: 0.8, sad: 0.1, angry: 0.05, surprise: 0.03, fear: 0.01, neutral: 0.01 },
    });
    expect(reading.emotion).toBe("happy");
    expect(reading.confidence).toBeCloseTo(0.8, 6);
  });

  it("falls back to neutral when all scores are missing or invalid", () => {
    const reading = normalizeReading("text", {
      index: 0,
      emotion: "happy",
      scores: { happy: Number.NaN, sad: undefined as unknown as number },
    });
    expect(reading.emotion).toBe("neutral");
    expect(reading.scores.neutral).toBe(1);
  });

  it("ignores unknown emotion labels", () => {
    const reading = normalizeReading("text", {
      index: 0,
      emotion: "ecstatic",
      scores: { happy: 0.6, sad: 0.2, angry: 0.1, surprise: 0.05, fear: 0.03, neutral: 0.02 },
    });
    expect(EMOTIONS.includes(reading.emotion as never)).toBe(true);
  });
});
