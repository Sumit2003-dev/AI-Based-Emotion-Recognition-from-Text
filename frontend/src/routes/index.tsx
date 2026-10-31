import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";

import { BatchAnalyzer } from "@/components/emotion/batch-analyzer";
import { SingleAnalyzer } from "@/components/emotion/single-analyzer";
import { RecentReadings, type HistoryEntry } from "@/components/emotion/recent-readings";
import type { EmotionReading } from "@/lib/emotion/emotions";

let entryCounter = 0;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Emotion Reader — read the feeling behind any text" },
      {
        name: "description",
        content:
          "Paste a tweet, message, or review and get its emotional reading — happy, sad, angry, surprise, fear, or neutral. Analyze one text or a whole batch.",
      },
      { property: "og:title", content: "Emotion Reader" },
      {
        property: "og:description",
        content:
          "Read the feeling behind any tweet, message, or review — one text or a whole batch.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  const [mode, setMode] = useState<"single" | "batch">("single");
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const addEntries = useCallback((readings: EmotionReading[]) => {
    const stamped = readings.map((reading) => ({
      ...reading,
      id: `r${++entryCounter}`,
      at: Date.now(),
    }));
    setHistory((prev) => [...stamped, ...prev]);
  }, []);

  const onResult = useCallback((reading: EmotionReading) => addEntries([reading]), [addEntries]);

  return (
    <div className="min-h-screen">
      <main className="mx-auto w-full max-w-5xl px-5 pb-20 pt-14 sm:px-8">
        <header>
          <p className="eyebrow text-primary">Emotion reader</p>
          <h1 className="mt-3 max-w-2xl font-display text-5xl leading-[1.05] tracking-tight sm:text-6xl">
            Read the feeling behind the words.
          </h1>
          <p className="mt-4 max-w-xl text-base leading-relaxed text-muted-foreground">
            Paste a tweet, a message, or a review — the reader weighs its wording, punctuation, and
            emoji against six emotions and shows you the balance.
          </p>
        </header>

        <section className="mt-10">
          <div className="flex gap-1 border-b border-border">
            {(
              [
                ["single", "One text"],
                ["batch", "A batch"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setMode(value)}
                className={`-mb-px border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                  mode === value
                    ? "border-primary text-foreground"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="mt-8">
            {mode === "single" ? (
              <SingleAnalyzer onResult={onResult} />
            ) : (
              <BatchAnalyzer onResults={addEntries} />
            )}
          </div>
        </section>

        <section className="mt-14 border-t border-border pt-8">
          <p className="eyebrow text-muted-foreground">Recent readings</p>
          <div className="mt-4">
            <RecentReadings history={history} />
          </div>
        </section>
      </main>
    </div>
  );
}
