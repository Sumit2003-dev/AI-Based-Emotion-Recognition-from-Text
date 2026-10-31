import { useState } from "react";

import { classifyEmotions } from "@/lib/emotion/classify.functions";
import {
  MAX_BATCH_TEXTS,
  MAX_TEXT_CHARS,
  topEmotionDistribution,
  type EmotionReading,
} from "@/lib/emotion/emotions";
import { EmotionBars, EmotionChip } from "./emotion-bars";

const BLURBS: Record<string, string> = {
  happy: "Joy, delight, satisfaction or amusement.",
  sad: "Loss, disappointment, grief or loneliness.",
  angry: "Irritation, outrage or frustration.",
  surprise: "Astonishment — pleasant or not.",
  fear: "Worry, anxiety or dread.",
  neutral: "Plainly stated, no strong feeling.",
};

export function BatchAnalyzer({ onResults }: { onResults: (readings: EmotionReading[]) => void }) {
  const [batchText, setBatchText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readings, setReadings] = useState<EmotionReading[] | null>(null);

  const lines = batchText
    .split("\n")
    .map((line) => line.trim().slice(0, MAX_TEXT_CHARS))
    .filter(Boolean)
    .slice(0, MAX_BATCH_TEXTS);
  const overflow = batchText.split("\n").filter((l) => l.trim()).length - lines.length;

  async function analyze() {
    if (lines.length === 0 || loading) return;
    setLoading(true);
    setError(null);
    try {
      const response = await classifyEmotions({ data: { texts: lines } });
      if (response.ok) {
        setReadings(response.readings);
        onResults(response.readings);
      } else {
        setError(response.error);
      }
    } catch {
      setError("The emotion engine couldn't be reached. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  const distribution = readings ? topEmotionDistribution(readings) : null;
  const distributionScores = distribution
    ? Object.fromEntries(
        Object.entries(distribution).map(([emotion, count]) => [
          emotion,
          readings!.length > 0 ? count / readings!.length : 0,
        ]),
      )
    : null;

  return (
    <div className="flex h-full flex-col gap-5">
      <div>
        <textarea
          value={batchText}
          onChange={(event) => setBatchText(event.target.value)}
          rows={8}
          placeholder={
            "One text per line — e.g.\nLoved it, five stars!\nRefund requested twice, still nothing.\nShipping took two days as promised."
          }
          className="w-full resize-none rounded-md border border-input bg-background px-4 py-3 font-sans text-sm leading-relaxed outline-none transition-colors placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30"
        />
        <p className="mt-1.5 text-xs text-muted-foreground">
          {lines.length} {lines.length === 1 ? "text" : "texts"} ready · up to {MAX_BATCH_TEXTS}{" "}
          lines
          {overflow > 0 && ` · ${overflow} more line${overflow === 1 ? "" : "s"} will be skipped`}
        </p>
      </div>

      <div className="mt-auto flex items-center gap-4">
        <button
          type="button"
          onClick={analyze}
          disabled={loading || lines.length === 0}
          className="inline-flex items-center justify-center rounded-md bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? `Reading ${lines.length}…` : "Read the batch"}
        </button>
        {error && <p className="text-sm text-destructive">{error}</p>}
      </div>

      {readings && readings.length > 0 && (
        <div className="border-t border-border pt-5">
          <div className="flex items-baseline justify-between gap-3">
            <p className="eyebrow text-muted-foreground">Batch readings</p>
            <p className="text-xs text-muted-foreground">{readings.length} analyzed</p>
          </div>
          <div className="mt-2">
            {readings.map((reading, i) => (
              <div
                key={i}
                className="flex items-center gap-3 border-b border-border py-2.5 last:border-0"
              >
                <span className="w-5 shrink-0 text-xs tabular-nums text-muted-foreground/70">
                  {i + 1}
                </span>
                <p className="min-w-0 flex-1 truncate text-sm">{reading.text}</p>
                <span className="hidden shrink-0 sm:block" title={BLURBS[reading.emotion]}>
                  <EmotionChip emotion={reading.emotion} confidence={reading.confidence} />
                </span>
              </div>
            ))}
          </div>
          {distributionScores && (
            <div className="mt-5">
              <p className="eyebrow text-muted-foreground">Batch temperament</p>
              <div className="mt-3">
                <EmotionBars scores={distributionScores as never} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
