import { EMOTION_META, EMOTIONS, type EmotionScores } from "@/lib/emotion/emotions";

function emotionColor(emotion: string) {
  return `var(--emotion-${emotion})`;
}

export function EmotionBars({ scores }: { scores: EmotionScores }) {
  const rows = EMOTIONS.map((emotion) => ({ emotion, value: scores[emotion] || 0 })).sort(
    (a, b) => b.value - a.value,
  );

  return (
    <div className="space-y-3">
      {rows.map(({ emotion, value }) => {
        const pct = Math.round(value * 100);
        const meta = EMOTION_META[emotion];
        const color = emotionColor(emotion);
        return (
          <div key={emotion} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2 min-w-0">
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="font-medium text-foreground text-xs truncate">
                  {meta?.label ?? emotion}
                </span>
              </div>
              <span className="shrink-0 font-semibold tabular-nums text-muted-foreground ml-2">
                {pct}%
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-secondary/50">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${Math.max(2, pct)}%`,
                  backgroundColor: color,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function EmotionChip({ emotion, confidence }: { emotion: string; confidence?: number }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-sm">
      <span
        className="h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: emotionColor(emotion) }}
      />
      <span className="font-medium">
        {EMOTION_META[emotion as keyof typeof EMOTION_META]?.label ?? emotion}
      </span>
      {confidence !== undefined && (
        <span className="text-xs tabular-nums text-muted-foreground">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </span>
  );
}
