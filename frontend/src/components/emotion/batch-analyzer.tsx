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
  happy: "Joy, delight, elation, triumph or satisfaction.",
  sad: "Loss, disappointment, grief or emotional fatigue.",
  angry: "Irritation, outrage, boundary violation or frustration.",
  fear: "Worry, apprehension, threat detection or anxiety.",
  surprise: "Astonishment, unexpected turns or sudden shock.",
  love: "Affection, deep warmth, romance, devotion or intimacy.",
  disgust: "Revulsion, aversion, foulness, moral recoil or contempt.",
  neutral: "Plainly stated, objective pragmatic communication.",
};

const SAMPLE_BATCH = [
  "I am so incredibly happy with this service! Best support ever 🎉",
  "My package was delivered damaged and two weeks late, terrible experience.",
  "Wait, are you telling me the event was cancelled today?!",
  "Please find the attached quarterly financial report for review.",
  "I am deeply worried about the upcoming surgery and cannot sleep.",
  "Never buying from this company again, absolute scam!",
  "मुझे यह जानकर बहुत खुशी हुई कि सब कुछ ठीक हो गया!",
].join("\n");

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

  function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      if (content) {
        // Simple CSV / newline parser: take first non-empty column or line
        const rows = content
          .split("\n")
          .map((r) => r.replace(/^["']|["']$/g, "").trim())
          .filter(Boolean)
          .slice(0, MAX_BATCH_TEXTS);
        setBatchText(rows.join("\n"));
      }
    };
    reader.readAsText(file);
  }

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
          rows={7}
          placeholder={
            "One text per line — e.g.\nLoved it, five stars! 🎉\nRefund requested twice, still nothing.\nShipping took two days as promised.\nमुझे आज बहुत खुशी हो रही है!"
          }
          className="w-full resize-none rounded-xl border border-input bg-card/60 p-4 font-sans text-sm leading-relaxed outline-none transition-all placeholder:text-muted-foreground/70 focus-visible:border-ring focus-visible:ring-2 focus-visible:ring-ring/30 shadow-sm"
        />
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <p>
            {lines.length} {lines.length === 1 ? "text" : "texts"} ready · up to {MAX_BATCH_TEXTS}{" "}
            lines
            {overflow > 0 && ` · ${overflow} more line${overflow === 1 ? "" : "s"} will be skipped`}
          </p>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setBatchText(SAMPLE_BATCH)}
              className="text-xs text-primary hover:underline"
            >
              Load Sample Batch
            </button>
            <span>•</span>
            <label className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
              <span>Import .CSV / .txt</span>
              <input
                type="file"
                accept=".csv,.txt"
                onChange={handleFileUpload}
                className="hidden"
              />
            </label>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <button
          type="button"
          onClick={analyze}
          disabled={loading || lines.length === 0}
          className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground shadow-sm transition-all hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? (
            <>
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent" />
              Reading {lines.length} texts…
            </>
          ) : (
            `Read batch (${lines.length})`
          )}
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
