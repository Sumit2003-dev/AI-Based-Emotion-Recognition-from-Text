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
        <footer className="mt-20 border-t border-border/70 pt-10 pb-12 space-y-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-primary" />
                <span className="font-display text-lg tracking-tight text-foreground">
                  Emotion Intelligence Lab
                </span>
              </div>
              <p className="text-xs text-muted-foreground max-w-md">
                Production-grade multi-class emotion recognition, affective computing, and
                psychometric sentiment analysis powered by PyTorch and FastAPI.
              </p>
            </div>

            {/* Social & Contact Links */}
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <a
                href="https://github.com/Sumit2003-dev"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 font-medium text-foreground hover:bg-secondary/70 hover:border-primary transition-all shadow-sm"
              >
                <svg className="h-3.5 w-3.5 fill-current" viewBox="0 0 24 24">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
                <span>GitHub</span>
              </a>

              <a
                href="https://www.linkedin.com/in/sumit2003-dev/"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 font-medium text-foreground hover:bg-secondary/70 hover:border-primary transition-all shadow-sm"
              >
                <svg className="h-3.5 w-3.5 fill-current text-[#0A66C2]" viewBox="0 0 24 24">
                  <path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.28 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.75M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z" />
                </svg>
                <span>LinkedIn</span>
              </a>

              <a
                href="mailto:sumit.ku5981@gmail.com"
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/60 px-3 py-1.5 font-medium text-foreground hover:bg-secondary/70 hover:border-primary transition-all shadow-sm"
              >
                <svg className="h-3.5 w-3.5 fill-none stroke-current stroke-2" viewBox="0 0 24 24">
                  <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                  <polyline points="22,6 12,13 2,6" />
                </svg>
                <span>Email</span>
              </a>

              <a
                href="https://github.com/Sumit2003-dev/AI-Based-Emotion-Recognition-from-Text"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-secondary/40 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
              >
                <span>Source Code</span>
                <span>↗</span>
              </a>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-between text-xs text-muted-foreground gap-3 border-t border-border/50 pt-6">
            <p className="flex items-center gap-1.5">
              <span>© {new Date().getFullYear()}</span>
              <strong className="font-semibold text-foreground">Sumit Kumar</strong>.
              <span>All rights reserved.</span>
            </p>
            <div className="flex items-center gap-3 text-[11px]">
              <span>FastAPI 0.115+</span>
              <span>•</span>
              <span>React 19 & TanStack</span>
              <span>•</span>
              <span>Hybrid Transformer Architecture</span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
}
