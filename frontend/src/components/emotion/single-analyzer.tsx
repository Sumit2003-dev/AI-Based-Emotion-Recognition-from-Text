import { useState } from "react";

import { classifyEmotions } from "@/lib/emotion/classify.functions";
import { MAX_TEXT_CHARS } from "@/lib/emotion/emotions";
import type { EmotionReading } from "@/lib/emotion/emotions";
import { EmotionBars } from "./emotion-bars";

const SAMPLES = [
  "Ugh, my flight got delayed THREE hours and they lost my bag. Never flying with this airline again.",
  "Just got the acceptance email!!! I literally screamed in the library 🎉",
  "The meeting is moved to 3pm on Thursday. Please confirm your attendance.",
  "Wait... they're announcing the results TODAY? I did not see that coming at all.",
  "I keep replaying the voicemail. I don't know how to tell them, my hands won't stop shaking.",
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
          placeholder="Paste a tweet, message, or review…"
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
            onClick={() => setText(sample)}
            disabled={loading}
            className="rounded-full border border-border px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-foreground disabled:opacity-50"
          >
            {
              [
                "Delayed flight ✈️",
                "Acceptance 🎉",
                "Meeting note",
                "Unexpected news",
                "The voicemail",
              ][i]
            }
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
