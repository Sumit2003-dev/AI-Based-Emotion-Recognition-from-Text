import { useState } from "react";
import { EMOTION_META, type Emotion, type EmotionReading } from "@/lib/emotion/emotions";
import { EmotionBars } from "./emotion-bars";

const EMOJI_MAP: Record<Emotion, string> = {
  happy: "😊",
  sad: "😢",
  angry: "😡",
  fear: "😨",
  surprise: "😲",
  love: "❤️",
  disgust: "🤢",
  neutral: "😐",
};

export function EmotionCockpit({
  reading,
  onReset,
}: {
  reading: EmotionReading | null;
  onReset?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  function copySuggestedReply(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  if (!reading) {
    return (
      <aside className="rounded-xl border border-border/80 bg-card/60 p-6 shadow-sm backdrop-blur-sm sm:p-7">
        <div className="flex items-center justify-between border-b border-border/60 pb-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-emerald-500" />
            </span>
            <span className="eyebrow text-muted-foreground">Neural Engine Ready</span>
          </div>
          <span className="rounded-md border border-border bg-secondary/50 px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
            FastAPI + Hybrid Model
          </span>
        </div>

        <div className="py-8 text-center sm:py-10">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-dashed border-border bg-secondary/30 text-3xl">
            🧠
          </div>
          <h2 className="mt-4 font-display text-2xl tracking-tight sm:text-3xl">
            Live Emotion Cockpit
          </h2>
          <p className="mx-auto mt-2 max-w-sm text-sm text-muted-foreground">
            Type or paste any message on the left, or select one of the multilingual sample chips to
            see real-time affective computing metrics.
          </p>
        </div>

        {/* Feature Preview Cards */}
        <div className="grid gap-3 pt-2 sm:grid-cols-2">
          <div className="rounded-lg border border-border/70 bg-background/50 p-3.5">
            <div className="text-base">🎭</div>
            <p className="mt-1 font-medium text-xs text-foreground">Mixed-Emotion Detection</p>
            <p className="mt-0.5 text-[11px] text-muted-foreground leading-snug">
              Captures complex emotional blends (e.g. hurt & disappointment beneath frustration).
            </p>
          </div>
          <div className="rounded-lg border border-border/70 bg-background/50 p-3.5">
            <div className="text-base">📊</div>
            <p className="mt-1 font-medium text-xs text-foreground">Estimated VAD Projections</p>
            <p className="mt-0.5 text-[11px] text-muted-foreground leading-snug">
              Psychometric mapping of Valence (Pleasure), Arousal (Energy), and Dominance (Agency).
            </p>
          </div>
          <div className="rounded-lg border border-border/70 bg-background/50 p-3.5">
            <div className="text-base">🌐</div>
            <p className="mt-1 font-medium text-xs text-foreground">Cross-Lingual & Hinglish</p>
            <p className="mt-0.5 text-[11px] text-muted-foreground leading-snug">
              Native Romanized Hinglish phonetic bridge with 100+ language neural translation.
            </p>
          </div>
          <div className="rounded-lg border border-border/70 bg-background/50 p-3.5">
            <div className="text-base">⚡</div>
            <p className="mt-1 font-medium text-xs text-foreground">Fast Latency (~25ms CPU)</p>
            <p className="mt-0.5 text-[11px] text-muted-foreground leading-snug">
              Optimized PyTorch CPU DistilRoBERTa pipeline with edge failover to Groq & Gemini.
            </p>
          </div>
        </div>
      </aside>
    );
  }

  const emotionColor = `var(--emotion-${reading.emotion})`;
  const emoji = EMOJI_MAP[reading.emotion] || "😐";
  const confidencePct = Math.round(reading.confidence * 100);
  const dim = reading.dimensions;
  const guidance = reading.guidance;

  return (
    <aside className="rounded-xl border border-border/80 bg-card p-6 shadow-sm sm:p-7 space-y-6">
      {/* Cockpit Top Bar */}
      <div className="flex items-center justify-between border-b border-border/60 pb-3.5">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: emotionColor }} />
          <span className="eyebrow text-muted-foreground">Active Intelligence Reading</span>
        </div>
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Clear
          </button>
        )}
      </div>

      {/* Hero Dominant Emotion Card */}
      <div
        className="relative overflow-hidden rounded-xl border p-5 sm:p-6 transition-all"
        style={{
          borderColor: `color-mix(in srgb, ${emotionColor} 35%, var(--border))`,
          background: `radial-gradient(circle at top right, color-mix(in srgb, ${emotionColor} 12%, transparent), transparent 70%), var(--card)`,
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <span className="eyebrow text-muted-foreground">Dominant Emotion</span>
            <div className="mt-1 flex items-baseline gap-3">
              <h2
                className="font-display text-4xl sm:text-5xl capitalize leading-none tracking-tight"
                style={{ color: emotionColor }}
              >
                {reading.emotion}
              </h2>
              <span className="rounded-full border border-border/80 bg-background/80 px-2.5 py-0.5 text-xs font-semibold tabular-nums text-foreground">
                {confidencePct}% confidence
              </span>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              {EMOTION_META[reading.emotion]?.blurb}
            </p>
          </div>
          <div className="text-5xl select-none sm:text-6xl animate-in zoom-in-50 duration-300">
            {emoji}
          </div>
        </div>

        {/* Multilingual / Translation Pill */}
        {reading.isMultilingual && reading.languageName && (
          <div className="mt-4 flex flex-wrap items-center gap-2 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2 text-xs text-primary">
            <span className="font-semibold">🌐 {reading.languageName}</span>
            {reading.translatedText && (
              <span className="text-muted-foreground">
                • Translated: &ldquo;{reading.translatedText}&rdquo;
              </span>
            )}
          </div>
        )}

        {/* Sentiment and Intensity Badges */}
        <div className="mt-4 flex flex-wrap items-center gap-2 text-xs">
          {reading.sentiment && (
            <span className="inline-flex items-center gap-1 rounded-md border border-border bg-secondary/50 px-2.5 py-1 text-foreground">
              <span className="text-muted-foreground">Sentiment:</span>
              <strong className="font-medium">{reading.sentiment}</strong>
              {reading.sentimentScore !== undefined && (
                <span className="text-[11px] text-muted-foreground">
                  ({reading.sentimentScore > 0 ? "+" : ""}
                  {reading.sentimentScore.toFixed(2)})
                </span>
              )}
            </span>
          )}
          {reading.intensityLevel && (
            <span className="inline-flex items-center gap-1 rounded-md border border-border bg-secondary/50 px-2.5 py-1 text-foreground">
              <span className="text-muted-foreground">Intensity:</span>
              <strong className="font-medium">{reading.intensityLevel}</strong>
            </span>
          )}
        </div>

        {/* Mixed-Emotion Detection & Nuanced Affect Blend */}
        {reading.isMixedEmotion && reading.secondaryEmotion && (
          <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs text-foreground">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-1.5 font-semibold text-amber-600 dark:text-amber-400">
                <span>🎭 Mixed Emotion Blend:</span>
                <span className="capitalize">
                  {reading.blendLabel || `${reading.emotion} + ${reading.secondaryEmotion}`}
                </span>
              </div>
              <span className="rounded-full border border-amber-500/30 bg-background/80 px-2 py-0.5 text-[10px] font-semibold tabular-nums text-muted-foreground">
                Co-occurring: {reading.secondaryEmotion} (
                {Math.round((reading.secondaryConfidence || 0) * 100)}%)
              </span>
            </div>
            {reading.blendExplanation && (
              <p className="mt-1.5 text-muted-foreground text-[11px] leading-relaxed">
                {reading.blendExplanation}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Emotion Spectrum Distribution */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <p className="eyebrow text-muted-foreground">Emotion Spectrum Distribution</p>
          <span className="text-xs text-muted-foreground">6 Ekman Spectrums (8-Class Neural Engine)</span>
        </div>
        <div className="rounded-lg border border-border/70 bg-secondary/20 p-4">
          <EmotionBars scores={reading.scores} />
        </div>
      </div>

      {/* Affective Computing (VAD Dimensions) */}
      {dim && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="eyebrow text-muted-foreground">
              Estimated Affective Dimensions (VAD Projection)
            </p>
            <span className="text-[10px] text-muted-foreground">Mehrabian-Russell Model</span>
          </div>
          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
            {/* Valence */}
            <div className="rounded-lg border border-border/70 bg-secondary/20 p-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Valence</span>
                <span className="font-medium tabular-nums text-foreground">
                  {dim.valence > 0 ? `+${dim.valence}` : dim.valence}
                </span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-primary"
                  style={{
                    width: `${Math.round(((dim.valence + 1) / 2) * 100)}%`,
                  }}
                />
              </div>
              <p className="mt-1.5 text-[10px] text-muted-foreground">
                {dim.valence > 0.1
                  ? "Pleasure / Positive"
                  : dim.valence < -0.1
                    ? "Pain / Negative"
                    : "Neutral Polarity"}
              </p>
            </div>

            {/* Arousal */}
            <div className="rounded-lg border border-border/70 bg-secondary/20 p-3">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Arousal</span>
                <span className="font-medium tabular-nums text-foreground">{dim.arousal}%</span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full bg-accent"
                  style={{ width: `${dim.arousal}%` }}
                />
              </div>
              <p className="mt-1.5 text-[10px] text-muted-foreground">{dim.arousal_label}</p>
            </div>

            {/* Dominance */}
            <div className="col-span-2 rounded-lg border border-border/70 bg-secondary/20 p-3 sm:col-span-1">
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Dominance</span>
                <span className="font-medium tabular-nums text-foreground">{dim.dominance}%</span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-secondary">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${dim.dominance}%`,
                    backgroundColor: "oklch(0.68 0.15 45)",
                  }}
                />
              </div>
              <p className="mt-1.5 text-[10px] text-muted-foreground">{dim.dominance_label}</p>
            </div>
          </div>
        </div>
      )}

      {/* Psychology & Empathy Guidance */}
      {guidance && (
        <div className="space-y-3 border-t border-border/60 pt-4">
          <p className="eyebrow text-muted-foreground">Cognitive Empathy & Advice</p>

          {guidance.psychology && (
            <div className="rounded-lg border border-border/70 bg-secondary/15 p-3.5 text-xs">
              <span className="font-semibold text-foreground">Psychological Context: </span>
              <span className="text-muted-foreground">{guidance.psychology}</span>
            </div>
          )}

          {guidance.empathy_tip && (
            <div className="rounded-lg border border-border/70 bg-secondary/15 p-3.5 text-xs">
              <span className="font-semibold text-foreground">Communication Tip: </span>
              <span className="text-muted-foreground">{guidance.empathy_tip}</span>
            </div>
          )}

          {guidance.suggested_reply && (
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-3.5 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-semibold text-primary">Suggested Response:</span>
                <button
                  type="button"
                  onClick={() => copySuggestedReply(guidance.suggested_reply!)}
                  className="rounded border border-primary/30 px-2 py-0.5 text-[11px] font-medium text-primary hover:bg-primary hover:text-primary-foreground transition-colors"
                >
                  {copied ? "Copied! ✓" : "Copy Reply"}
                </button>
              </div>
              <p className="mt-1.5 italic text-muted-foreground">
                &ldquo;{guidance.suggested_reply}&rdquo;
              </p>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
