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

  it("preserves LLM suggested_reply and guidance in toReadings", () => {
    const readings = toReadings(
      [
        {
          index: 1,
          emotion: "angry",
          scores: {
            happy: 0.01,
            sad: 0.09,
            angry: 0.85,
            surprise: 0.02,
            fear: 0.01,
            neutral: 0.02,
          },
          suggested_reply: "Tension mat le bhai, sab sort ho jayega.",
          psychology: "Frustration due to exam pressure.",
          empathy_tip: "Stay calm and supportive.",
        },
      ],
      ["kal exam hai darr lag raha hai"],
    );
    expect(readings[0]?.guidance?.suggested_reply).toBe("Tension mat le bhai, sab sort ho jayega.");
    expect(readings[0]?.guidance?.psychology).toBe("Frustration due to exam pressure.");
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
    expect(result.scores.angry).toBeGreaterThan(0.6);
    expect(result.scores.sad).toBeLessThan(0.15);
    expect(result.isMultilingual).toBe(true);
    expect(result.translatedText?.toLowerCase()).toContain("mind");
  });

  it("accurately classifies Hinglish disbelief / frustration sentence", async () => {
    const result = await classifyLocalText("pagal h kya bhai");
    expect(["angry", "surprise"]).toContain(result.dominantEmotion);
  });

  it("correctly identifies non-English Hindi devanagari input", async () => {
    const result = await classifyLocalText("मुझे बहुत गुस्सा आ रहा है");
    expect(result.dominantEmotion).toBe("angry");
    expect(result.scores.angry).toBeGreaterThan(0.5);
    expect(result.detectedLanguage).toBe("hi");
    expect(result.isMultilingual).toBe(true);
  });

  it("handles contrastive conjunctions shifting to sadness", async () => {
    const result = await classifyLocalText(
      "maine itna accha presentation diya lekin sabne reject kar diya",
    );
    expect(result.dominantEmotion).toBe("sad");
    expect(result.scores.sad).toBeGreaterThan(0.4);
    expect(result.scores.happy).toBeLessThan(0.15);
  });

  it("passes all 7 benchmark sentences for advanced Hinglish and contrast regression", async () => {
    // 1. Meri mehnat rang layi -> Happy
    const r1 = await classifyLocalText("Aaj meri mehnat rang layi!");
    expect(r1.dominantEmotion).toBe("happy");
    expect(r1.scores.happy).toBeGreaterThan(0.5);

    // 2. Ab kya fayda -> Sad
    const r2 = await classifyLocalText("Ab kya fayda bolne ka");
    expect(r2.dominantEmotion).toBe("sad");
    expect(r2.scores.sad).toBeGreaterThan(0.5);

    // 3. Andar se thak chuka hoon -> Sad
    const r3 = await classifyLocalText("Andar se thak chuka hoon");
    expect(r3.dominantEmotion).toBe("sad");
    expect(r3.scores.sad).toBeGreaterThan(0.4);

    // 4. Sabke saamne smile karta hoon, par andar se thak chuka hoon -> Sad (smile does not override exhaustion)
    const r4 = await classifyLocalText(
      "Sabke saamne smile karta hoon, par andar se thak chuka hoon",
    );
    expect(r4.dominantEmotion).toBe("sad");
    expect(r4.scores.happy).toBeLessThan(0.15);

    // 5. Oh wow, another deadline! Exactly what I needed. -> Angry (sarcasm)
    const r5 = await classifyLocalText("Oh wow, another deadline! Exactly what I needed.");
    expect(r5.dominantEmotion).toBe("angry");

    // 6. Feeling pointless -> Sad
    const r6 = await classifyLocalText("Feeling pointless");
    expect(r6.dominantEmotion).toBe("sad");

    // 7. Hiding exhaustion -> Sad
    const r7 = await classifyLocalText("Hiding exhaustion");
    expect(r7.dominantEmotion).toBe("sad");
  });

  it("passes all 8 sentences of the official regression test checklist", async () => {
    // Test 1: Are yaar, itna acha kaam kiya fir bhi dant padi. -> Sadness / Anger
    const t1 = await classifyLocalText("Are yaar, itna acha kaam kiya fir bhi dant padi.");
    expect(["sad", "angry"]).toContain(t1.dominantEmotion);

    // Test 2: Yaar, aaj finally meri mehnat rang laayi! -> Happiness
    const t2 = await classifyLocalText("Yaar, aaj finally meri mehnat rang laayi!");
    expect(t2.dominantEmotion).toBe("happy");

    // Test 3: Mujhe kisi se koi umeed nahi rahi ab. -> Sadness
    const t3 = await classifyLocalText("Mujhe kisi se koi umeed nahi rahi ab.");
    expect(t3.dominantEmotion).toBe("sad");

    // Test 4: Oh wow, another deadline! Exactly what I needed. -> Anger (sarcasm)
    const t4 = await classifyLocalText("Oh wow, another deadline! Exactly what I needed.");
    expect(t4.dominantEmotion).toBe("angry");

    // Test 5: Phone table par rakha hai aur main khush hoon. -> Happiness (locative par does not trigger contrast)
    const t5 = await classifyLocalText("Phone table par rakha hai aur main khush hoon.");
    expect(t5.dominantEmotion).toBe("happy");

    // Test 6: Maine bahut mehnat ki, lekin mujhe promotion mil gaya! -> Happiness (positive contrast outcome)
    const t6 = await classifyLocalText("Maine bahut mehnat ki, lekin mujhe promotion mil gaya!");
    expect(t6.dominantEmotion).toBe("happy");

    // Test 7: Sabke saamne hasta hoon, par andar se toot chuka hoon. -> Sadness (concealed heartbreak)
    const t7 = await classifyLocalText("Sabke saamne hasta hoon, par andar se toot chuka hoon.");
    expect(t7.dominantEmotion).toBe("sad");

    // Test 8: Mujhe daant nahi padi, main bas mazaak kar raha tha. -> Neutral / Happy
    const t8 = await classifyLocalText("Mujhe daant nahi padi, main bas mazaak kar raha tha.");
    expect(["neutral", "happy"]).toContain(t8.dominantEmotion);
  });

  it("passes all 10 diagnostic evaluation sentences with high precision", async () => {
    // 1. Bas ab bahut ho gaya. -> Anger
    const d1 = await classifyLocalText("Bas ab bahut ho gaya.");
    expect(d1.dominantEmotion).toBe("angry");

    // 2. Mujhe tumhari baatein nahi sunni. -> Anger
    const d2 = await classifyLocalText("Mujhe tumhari baatein nahi sunni.");
    expect(d2.dominantEmotion).toBe("angry");

    // 3. Tumhare bina meri life adhoori hai. -> Happy (Love spectrum)
    const d3 = await classifyLocalText("Tumhare bina meri life adhoori hai.");
    expect(d3.dominantEmotion).toBe("happy");

    // 4. I was worried earlier, but now everything feels fine. -> Happy (Relief / Resolution)
    const d4 = await classifyLocalText("I was worried earlier, but now everything feels fine.");
    expect(d4.dominantEmotion).toBe("happy");

    // 5. I received the job offer, but I am scared of moving away from home. -> Fear dominant, Happy secondary, Mixed True
    const d5 = await classifyLocalText(
      "I received the job offer, but I am scared of moving away from home.",
    );
    expect(d5.dominantEmotion).toBe("fear");
    expect(d5.secondaryEmotion).toBe("happy");
    expect(d5.isMixed).toBe(true);

    // 6. I feel so hopeless and everything is falling apart. -> Sadness
    const d6 = await classifyLocalText("I feel so hopeless and everything is falling apart.");
    expect(d6.dominantEmotion).toBe("sad");

    // 7. This is completely unacceptable and I demand a full refund immediately! -> Anger
    const d7 = await classifyLocalText(
      "This is completely unacceptable and I demand a full refund immediately!",
    );
    expect(d7.dominantEmotion).toBe("angry");

    // 8. I cannot stop smiling after hearing the wonderful news today! -> Happy (Joy)
    const d8 = await classifyLocalText(
      "I cannot stop smiling after hearing the wonderful news today!",
    );
    expect(d8.dominantEmotion).toBe("happy");

    // 9. My hands won't stop shaking and I can barely breathe from the panic. -> Fear
    const d9 = await classifyLocalText(
      "My hands won't stop shaking and I can barely breathe from the panic.",
    );
    expect(d9.dominantEmotion).toBe("fear");

    // 10. The rotten garbage was covered in crawling maggots and smelled putrid. -> Angry (Disgust/Aversion spectrum)
    const d10 = await classifyLocalText(
      "The rotten garbage was covered in crawling maggots and smelled putrid.",
    );
    expect(["angry", "sad"]).toContain(d10.dominantEmotion);
  });
});
