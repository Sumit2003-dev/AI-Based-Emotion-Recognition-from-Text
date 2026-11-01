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
  it("accurately classifies Hinglish anger sentence", async () => {
    const result = await classifyLocalText("mujhe gussa aa raha h bata raha hu");
    expect(result.dominantEmotion).toBe("angry");
    expect(result.scores.angry).toBeGreaterThan(0.5);
    expect(["hi", "hinglish"]).toContain(result.detectedLanguage);
    expect(result.isMultilingual).toBe(true);
    expect(result.tokens.some((t) => t.token.includes("gussa") && t.is_emotional)).toBe(true);
  });

  it("accurately classifies 'bhai deemag mat kharab kr' as angry with coherent translation", async () => {
    const result = await classifyLocalText("bhai deemag mat kharab kr");
    expect(result.dominantEmotion).toBe("angry");
    expect(result.scores.angry).toBeGreaterThan(0.7);
    expect(result.scores.sad).toBeLessThan(0.15);
    expect(result.isMultilingual).toBe(true);
    expect(result.translatedText?.toLowerCase()).toContain("mind");
  });

  it("accurately classifies Hinglish disbelief / frustration sentence", async () => {
    const result = await classifyLocalText("pagal h kya bhai");
    expect(["angry", "surprise"]).toContain(result.dominantEmotion);
    expect(result.isMultilingual).toBe(true);
    expect(result.scores.neutral).toBeLessThan(0.4);
  });

  it("accurately classifies Hinglish joy sentence", async () => {
    const result = await classifyLocalText("bhai bohot mast film thi");
    expect(result.dominantEmotion).toBe("happy");
    expect(result.scores.happy).toBeGreaterThan(0.6);
    expect(result.isMultilingual).toBe(true);
    expect(result.translatedText).toBeTruthy();
  });

  it("accurately classifies Hinglish fear sentence", async () => {
    const result = await classifyLocalText("bohot darr lag raha hai mujhe");
    expect(result.dominantEmotion).toBe("fear");
    expect(result.scores.fear).toBeGreaterThan(0.6);
    expect(result.isMultilingual).toBe(true);
  });

  it("accurately classifies Hinglish sadness sentence", async () => {
    const result = await classifyLocalText("bohot dukh hua sunkar");
    expect(result.dominantEmotion).toBe("sad");
    expect(result.scores.sad).toBeGreaterThan(0.6);
    expect(result.isMultilingual).toBe(true);
  });

  it("accurately classifies negation 'i dont like you' as angry and not happy", async () => {
    const result = await classifyLocalText("i dont like you");
    expect(result.dominantEmotion).toBe("angry");
    expect(result.scores.happy).toBeLessThan(0.1);
    expect(result.scores.angry).toBeGreaterThan(0.5);
  });

  it("accurately classifies 'i am not happy' as sad and not happy", async () => {
    const result = await classifyLocalText("i am not happy");
    expect(result.dominantEmotion).toBe("sad");
    expect(result.scores.happy).toBeLessThan(0.1);
  });

  it("accurately classifies Hindi joy sentence", async () => {
    const result = await classifyLocalText("आज मैं बहुत खुश हूँ");
    expect(result.dominantEmotion).toBe("happy");
    expect(result.detectedLanguage).toBe("hi");
  });
});
