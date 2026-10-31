import { useState } from "react";

import { classifyEmotions } from "@/lib/emotion/classify.functions";
import { MAX_TEXT_CHARS } from "@/lib/emotion/emotions";
import type { EmotionReading } from "@/lib/emotion/emotions";
import { EmotionBars } from "./emotion-bars";

const SAMPLES = [
  {
    label: "Delayed flight ✈️",
    text: "Ugh, my flight got delayed THREE hours and they lost my bag. Never flying with this airline again.",
  },
  {
    label: "Acceptance 🎉",
    text: "Just got the acceptance email!!! I literally screamed in the library 🎉",
  },
  {
    label: "Hindi 🇮🇳",
    text: "मुझे आज बहुत खुशी हो रही है! परीक्षा में पहला स्थान मिला 🎉",
  },
  {
    label: "Spanish 🇪🇸",
    text: "Estoy extremadamente enojado con este servicio tan pésimo y tardío.",
  },
  {
    label: "French 🇫🇷",
    text: "Je suis tellement triste et seul depuis ton départ.",
  },
  {
    label: "Hinglish 🇮🇳",
    text: "mujhe bohot zyada gussa aa raha hai is baat par",
  },
  {
    label: "Meeting note",
    text: "The meeting is moved to 3pm on Thursday. Please confirm your attendance.",
  },
];

export function SingleAnalyzer({ onResult }: { onResult: (reading: EmotionReading) => void }) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reading, setReading] = useState<EmotionReading | null>(null);

  async function analyze() {
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    setLoading(true);
    setError(null);
    try {
      const response = await classifyEmotions({ data: { texts: [trimmed] } });
      if (response.ok) {
        const first = response.readings[0];
        if (first) {
          setReading(first);
          onResult(first);
        } else {
          setError("The engine returned no reading. Try again.");
        }
      } else {
        setError(response.error);
      }
    } catch {
      setError("The emotion engine couldn't be reached. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col gap-5">
      <div className="relative">
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          maxLength={MAX_TEXT_CHARS}
          rows={5}
          placeholder="Paste a tweet, message, or review in English, Hindi, Spanish, French, or any language…"
          className="w-full resize-none rounded-md border border-input bg-background px-4 py-3 font-sans text-base leading-relaxed outline-none transition-colors placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
        />
        <span className="absolute bottom-2.5 right-3 text-[11px] tabular-nums text-muted-foreground/70">
          {text.length}/{MAX_TEXT_CHARS}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow text-muted-foreground">Try</span>
        {SAMPLES.map((sample, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setText(sample.text)}
            disabled={loading}
            className="rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground disabled:opacity-50"
          >
            {sample.label}
          </button>
        ))}
      </div>

      <div className="mt-auto flex items-center gap-4">
        <button
          type="button"
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Reading…" : "Read the emotion"}
        </button>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>

      {reading && (
        <div className="border-t border-border pt-5">
          <p className="eyebrow text-muted-foreground">Reading</p>
          <div className="mt-2 flex items-baseline gap-3">
            <span
              className="font-display text-5xl italic leading-none"
              style={{ color: `var(--emotion-${reading.emotion})` }}
            >
              {reading.emotion}
            </span>
            <span className="text-sm tabular-nums text-muted-foreground">
              {Math.round(reading.confidence * 100)}% confidence
            </span>
          </div>

          {reading.isMultilingual && reading.languageName && (
            <div className="mt-2.5 inline-flex flex-wrap items-center gap-1.5 rounded-md border border-primary/20 bg-primary/5 px-2.5 py-1 text-xs text-primary">
              <span className="font-semibold">🌐 {reading.languageName}</span>
              {reading.translatedText && (
                <span className="text-muted-foreground">• En: "{reading.translatedText}"</span>
              )}
            </div>
          )}

          <p className="mt-2 text-sm text-muted-foreground">
            {reading.emotion === "neutral"
              ? "Plainly stated, no strong feeling."
              : reading.emotion === "happy"
                ? "Joy, delight, satisfaction or amusement."
                : reading.emotion === "sad"
                  ? "Loss, disappointment, grief or loneliness."
                  : reading.emotion === "angry"
                    ? "Irritation, outrage or frustration."
                    : reading.emotion === "surprise"
                      ? "Astonishment — pleasant or not."
                      : "Worry, anxiety or dread."}
          </p>
          <div className="mt-5">
            <EmotionBars scores={reading.scores} />
          </div>
        </div>
      )}
    </div>
  );
}
