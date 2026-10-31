import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";

import { BatchAnalyzer } from "@/components/emotion/batch-analyzer";
import { SingleAnalyzer } from "@/components/emotion/single-analyzer";
import { EmotionCockpit } from "@/components/emotion/emotion-cockpit";
import { RecentReadings, type HistoryEntry } from "@/components/emotion/recent-readings";
import type { EmotionReading } from "@/lib/emotion/emotions";

let entryCounter = 0;

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Emotion Reader — Multilingual AI Emotion & Sentiment Recognition" },
      {
        name: "description",
        content:
          "Analyze emotions, sentiment, psychometric VAD dimensions, and empathy guidance in 100+ languages using a Hybrid Neural Transformer.",
      },
      { property: "og:title", content: "Emotion Reader — Multilingual AI Emotion Recognition" },
      {
        property: "og:description",
        content:
          "Read the feeling behind any tweet, message, or review — supported in English, Hindi, Spanish, French, and 100+ languages.",
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
  const [activeReading, setActiveReading] = useState<EmotionReading | null>(null);

  const addEntries = useCallback((readings: EmotionReading[]) => {
    const stamped = readings.map((reading) => ({
      ...reading,
      id: `r${++entryCounter}`,
      at: Date.now(),
    }));
    setHistory((prev) => [...stamped, ...prev]);
    if (readings[0]) {
      setActiveReading(readings[0]);
    }
  }, []);

  const onResult = useCallback(
    (reading: EmotionReading) => {
      setActiveReading(reading);
      addEntries([reading]);
    },
    [addEntries],
  );

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto w-full max-w-7xl 2xl:max-w-[1550px] px-4 sm:px-6 lg:px-8 xl:px-10 pb-20 pt-8 sm:pt-12">
        {/* Top Branding & Meta Status Bar */}
        <header className="border-b border-border/70 pb-6 mb-8">
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-primary" />
              <p className="eyebrow text-primary">Emotion Intelligence Lab</p>
              <span className="text-muted-foreground">•</span>
              <span className="text-muted-foreground">Affective Neural System</span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full border border-primary/25 bg-primary/10 px-2.5 py-0.5 text-[11px] font-medium text-primary">
                🌐 100+ Languages
              </span>
              <span className="rounded-full border border-border bg-secondary/50 px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground">
                VAD Affective Computing
              </span>
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                Engine Active
              </span>
            </div>
          </div>

          <div className="mt-4 max-w-3xl">
            <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl leading-[1.06] tracking-tight text-foreground">
              Read the feeling behind the words.
            </h1>
            <p className="mt-3 text-base sm:text-lg leading-relaxed text-muted-foreground max-w-2xl">
              Paste a tweet, customer review, or personal message in any language. Our neural
              classifier evaluates wording, emoji, and semantic cues across six core emotions, VAD
              dimensions, and empathetic response strategies.
            </p>
          </div>
        </header>

        {/* Responsive Desktop Two-Column Cockpit Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-10 xl:gap-12 items-start">
          {/* Left Column: Input, Tabs, Samples, & Session History (7 cols) */}
          <div className="lg:col-span-7 space-y-8">
            <section className="rounded-2xl border border-border/80 bg-card/40 p-5 sm:p-7 shadow-sm backdrop-blur-sm">
              <div className="flex items-center justify-between border-b border-border/60 pb-4 mb-6">
                <div className="flex gap-2">
                  {(
                    [
                      ["single", "Single Text Analysis"],
                      ["batch", "Batch Dataset / CSV"],
                    ] as const
                  ).map(([value, label]) => (
                    <button
                      key={value}
                      type="button"
                      onClick={() => setMode(value)}
                      className={`rounded-lg px-3.5 py-1.5 text-xs font-medium transition-all ${
                        mode === value
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground hover:bg-secondary/60"
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <span className="hidden sm:inline-block text-[11px] text-muted-foreground">
                  {mode === "single" ? "Instant analysis" : "Up to 40 items"}
                </span>
              </div>

              <div>
                {mode === "single" ? (
                  <SingleAnalyzer onResult={onResult} activeReading={activeReading} />
                ) : (
                  <BatchAnalyzer onResults={addEntries} />
                )}
              </div>
            </section>

            {/* Session History Feed */}
            <section className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="eyebrow text-muted-foreground">Session History</p>
                  <h3 className="text-sm font-medium text-foreground">
                    Recently Analyzed Sentences
                  </h3>
                </div>
                {history.length > 0 && (
                  <button
                    type="button"
                    onClick={() => setHistory([])}
                    className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    Clear history
                  </button>
                )}
              </div>
              <RecentReadings
                history={history}
                onSelect={(entry) => setActiveReading(entry)}
                activeId={
                  activeReading?.text
                    ? history.find((h) => h.text === activeReading.text)?.id
                    : undefined
                }
              />
            </section>
          </div>

          {/* Right Column: Live Emotion Cockpit (5 cols, Sticky on desktop) */}
          <div className="lg:col-span-5 lg:sticky lg:top-6">
            <EmotionCockpit reading={activeReading} onReset={() => setActiveReading(null)} />
          </div>
        </div>

        {/* Global Footer */}
        <footer className="mt-20 border-t border-border/70 pt-8 pb-12 flex flex-col sm:flex-row items-center justify-between text-xs text-muted-foreground gap-4">
          <p>
            Designed & Developed by{" "}
            <strong className="font-semibold text-foreground">Sumit Kumar</strong>
          </p>
          <div className="flex flex-wrap items-center gap-4">
            <a
              href="https://github.com/Sumit2003-dev/AI-Based-Emotion-Recognition-from-Text"
              target="_blank"
              rel="noreferrer"
              className="hover:text-foreground transition-colors underline underline-offset-2"
            >
              GitHub Repository
            </a>
            <span>•</span>
            <span>FastAPI Backend</span>
            <span>•</span>
            <span>Hybrid Transformer Model</span>
          </div>
        </footer>
      </main>
    </div>
  );
}
