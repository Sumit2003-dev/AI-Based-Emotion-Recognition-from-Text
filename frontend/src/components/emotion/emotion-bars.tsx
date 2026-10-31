import { EMOTION_META, EMOTIONS, type EmotionScores } from "@/lib/emotion/emotions";

function emotionColor(emotion: string) {
  return `var(--emotion-${emotion})`;
}

export function EmotionBars({ scores }: { scores: EmotionScores }) {
  const rows = EMOTIONS.map((emotion) => ({ emotion, value: scores[emotion] })).sort(
    (a, b) => b.value - a.value,
  );

  return (
    <div className="space-y-2.5">
      {rows.map(({ emotion, value }) => (
        <div key={emotion} className="flex items-center gap-3">
          <span className="eyebrow w-20 shrink-0 text-muted-foreground">
            {EMOTION_META[emotion].label}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${Math.max(2, Math.round(value * 100))}%`,
                backgroundColor: emotionColor(emotion),
              }}
            />
          </div>
          <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
            {Math.round(value * 100)}%
          </span>
        </div>
      ))}
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
