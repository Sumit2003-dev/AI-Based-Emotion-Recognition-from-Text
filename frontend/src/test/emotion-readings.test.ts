import { describe, expect, it } from "vitest";

import { classifyLocalText, toReadings } from "@/lib/emotion/classify.functions";

const texts = ["Loved it, five stars!", "This is the third time support ignored me."];

describe("toReadings", () => {
  it("maps 1-based result indexes back onto the right texts", () => {
    const readings = toReadings(
      [
        {
          index: 1,
          emotion: "happy",
          scores: { happy: 0.9, sad: 0.02, angry: 0.02, surprise: 0.04, fear: 0.01, neutral: 0.01 },
        },
        {
          index: 2,
          emotion: "angry",
          scores: {
            happy: 0.01,
            sad: 0.14,
            angry: 0.77,
            surprise: 0.02,
            fear: 0.02,
            neutral: 0.04,
          },
        },
      ],
      texts,
    );
    const [first, second] = readings;
    expect(readings).toHaveLength(2);
    expect(first?.text).toBe(texts[0]);
    expect(first?.emotion).toBe("happy");
    expect(second?.text).toBe(texts[1]);
    expect(second?.emotion).toBe("angry");
  });

  it("drops results whose index is out of range", () => {
    const readings = toReadings(
      [
        {
          index: 5,
          emotion: "happy",
          scores: { happy: 1, sad: 0, angry: 0, surprise: 0, fear: 0, neutral: 0 },
        },
      ],
      texts,
    );
    expect(readings).toHaveLength(0);
  });
});

describe("classifyLocalText", () => {
  it("accurately classifies Hinglish anger sentence", () => {
    const result = classifyLocalText("mujhe gussa aa raha h bata raha hu");
    expect(result.dominantEmotion).toBe("angry");
    expect(result.scores.angry).toBeGreaterThan(0.5);
    expect(result.detectedLanguage).toBe("hinglish");
    expect(result.isMultilingual).toBe(true);
    expect(result.tokens.some((t) => t.token.includes("gussa") && t.is_emotional)).toBe(true);
  });

  it("accurately classifies Hinglish disbelief / frustration sentence", () => {
    const result = classifyLocalText("pagal h kya bhai");
    expect(["angry", "surprise"]).toContain(result.dominantEmotion);
    expect(result.detectedLanguage).toBe("hinglish");
    expect(result.scores.neutral).toBeLessThan(0.3);
  });

  it("accurately classifies Hindi joy sentence", () => {
    const result = classifyLocalText("आज मैं बहुत खुश हूँ");
    expect(result.dominantEmotion).toBe("happy");
    expect(result.detectedLanguage).toBe("hi");
  });
});
