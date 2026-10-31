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

export function SingleAnalyzer({
  onResult,
  activeReading,
}: {
  onResult: (reading: EmotionReading) => void;
  activeReading?: EmotionReading | null;
}) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reading, setReading] = useState<EmotionReading | null>(activeReading || null);

  // Sync when parent changes activeReading (e.g. clicked from history)
  if (activeReading && activeReading !== reading && activeReading.text !== text) {
    setReading(activeReading);
    setText(activeReading.text);
  }

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

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      analyze();
    }
  };

  const emotionalTokens = reading?.tokens?.filter((t) => t.is_emotional) || [];

  return (
    <div className="flex h-full flex-col gap-5">
      <div className="relative">
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={handleKeyDown}
          maxLength={MAX_TEXT_CHARS}
          rows={5}
          placeholder="Paste a tweet, message, or review in English, Hindi, Spanish, French, or any language… (Ctrl + Enter to read)"
          className="w-full resize-none rounded-xl border border-input bg-card/60 p-4 font-sans text-base leading-relaxed outline-none transition-all placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 shadow-sm"
        />
        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          {text.length > 0 && (
            <button
              type="button"
              onClick={() => {
                setText("");
                setReading(null);
              }}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors mr-1"
            >
              Clear
            </button>
          )}
          <span className="text-[11px] tabular-nums text-muted-foreground/70 bg-secondary/80 px-1.5 py-0.5 rounded">
            {text.length}/{MAX_TEXT_CHARS}
          </span>
        </div>
      </div>

      {/* Preset Chips */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="eyebrow text-muted-foreground">Try</span>
        {SAMPLES.map((sample, i) => (
          <button
            key={i}
            type="button"
            onClick={() => setText(sample.text)}
            disabled={loading}
            className="rounded-full border border-border bg-card/40 px-3 py-1 text-xs text-muted-foreground transition-all hover:border-primary hover:text-foreground hover:bg-card disabled:opacity-50"
          >
            {sample.label}
          </button>
        ))}
      </div>

      {/* Action Bar */}
      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={analyze}
          disabled={loading || !text.trim()}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground shadow-sm transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
              Reading emotion…
            </>
          ) : (
            <>
              <span>Read the emotion</span>
              <kbd className="hidden sm:inline-block rounded bg-primary-foreground/20 px-1.5 py-0.5 text-[10px] uppercase font-sans">
                ⌘↵
              </kbd>
            </>
          )}
        </button>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>

      {/* Token Highlights (if words were identified as emotional cues) */}
      {emotionalTokens.length > 0 && (
        <div className="rounded-xl border border-border/70 bg-card/40 p-3.5">
          <p className="eyebrow text-muted-foreground">Emotional Semantic Cues</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {emotionalTokens.map((t, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1 rounded-md border border-border/70 px-2 py-0.5 text-xs font-medium"
                style={{
                  backgroundColor: `color-mix(in srgb, var(--emotion-${t.emotion}) 15%, transparent)`,
                  borderColor: `color-mix(in srgb, var(--emotion-${t.emotion}) 40%, transparent)`,
                  color: `var(--emotion-${t.emotion})`,
                }}
              >
                <span>{t.token}</span>
                <span className="text-[10px] opacity-75 capitalize">({t.emotion})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
