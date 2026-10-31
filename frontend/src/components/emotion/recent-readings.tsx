import { EMOTION_META, topEmotionDistribution, type EmotionReading } from "@/lib/emotion/emotions";
import { EmotionChip } from "./emotion-bars";

export type HistoryEntry = EmotionReading & { id: string; at: number };

function formatTime(at: number) {
  return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function RecentReadings({ history }: { history: HistoryEntry[] }) {
  if (history.length === 0) {
    return (
      <p className="py-6 text-sm text-muted-foreground">
        Nothing read yet. Single and batch results collect here for this visit only — they clear
        when you leave.
      </p>
    );
  }

  const distribution = topEmotionDistribution(history);
  const total = history.length;

  return (
    <div className="grid gap-8 md:grid-cols-[1fr_260px]">
      <div>
        {history.map((entry) => (
          <div
            key={entry.id}
            className="flex items-center gap-3 border-b border-border py-2.5 last:border-0"
          >
            <span className="w-12 shrink-0 text-xs tabular-nums text-muted-foreground/70">
              {formatTime(entry.at)}
            </span>
            <p className="min-w-0 flex-1 truncate text-sm">{entry.text}</p>
            <span className="shrink-0">
              <EmotionChip emotion={entry.emotion} confidence={entry.confidence} />
            </span>
          </div>
        ))}
      </div>
      <div>
        <p className="eyebrow text-muted-foreground">Session temperament</p>
        <div className="mt-3 space-y-2">
          {Object.entries(distribution)
            .sort((a, b) => b[1] - a[1])
            .map(([emotion, count]) => (
              <div key={emotion} className="flex items-center gap-3">
                <span className="eyebrow w-20 shrink-0 text-muted-foreground">
                  {EMOTION_META[emotion as keyof typeof EMOTION_META].label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max(count > 0 ? 6 : 0, Math.round((count / total) * 100))}%`,
                      backgroundColor: `var(--emotion-${emotion})`,
                    }}
                  />
                </div>
                <span className="w-8 text-right text-xs tabular-nums text-muted-foreground">
                  {count}
                </span>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
