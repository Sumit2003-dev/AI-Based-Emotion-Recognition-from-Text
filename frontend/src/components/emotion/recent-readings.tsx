import { EMOTION_META, topEmotionDistribution, type EmotionReading } from "@/lib/emotion/emotions";
import { EmotionChip } from "./emotion-bars";

export type HistoryEntry = EmotionReading & { id: string; at: number };

function formatTime(at: number) {
  return new Date(at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function RecentReadings({
  history,
  onSelect,
  activeId,
}: {
  history: HistoryEntry[];
  onSelect?: (entry: HistoryEntry) => void;
  activeId?: string;
}) {
  if (history.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/80 p-6 text-center text-sm text-muted-foreground">
        <p>No readings recorded yet.</p>
        <p className="mt-1 text-xs text-muted-foreground/80">
          Your analyzed sentences will collect here automatically for this session.
        </p>
      </div>
    );
  }

  const distribution = topEmotionDistribution(history);
  const total = history.length;

  return (
    <div className="grid gap-8 lg:grid-cols-12 items-start">
      <div className="lg:col-span-7 divide-y divide-border/60 rounded-xl border border-border/80 bg-card/40 p-2 overflow-hidden">
        <div className="flex items-center justify-between px-3 py-2 text-xs text-muted-foreground">
          <span>
            {total} {total === 1 ? "entry" : "entries"} recorded
          </span>
          <span className="text-[11px]">Click any item to inspect in Cockpit</span>
        </div>
        <div className="max-h-[360px] overflow-y-auto divide-y divide-border/40 pr-1">
          {history.map((entry) => {
            const isActive = activeId === entry.id;
            return (
              <button
                key={entry.id}
                type="button"
                onClick={() => onSelect?.(entry)}
                className={`flex w-full items-center gap-3 px-3 py-2.5 text-left transition-colors rounded-lg ${
                  isActive ? "bg-primary/10 border border-primary/30" : "hover:bg-secondary/60"
                }`}
              >
                <span className="w-14 shrink-0 text-xs tabular-nums text-muted-foreground/70">
                  {formatTime(entry.at)}
                </span>
                <p className="min-w-0 flex-1 truncate text-sm font-sans text-foreground">
                  {entry.text}
                </p>
                <span className="shrink-0">
                  <EmotionChip emotion={entry.emotion} confidence={entry.confidence} />
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="lg:col-span-5 rounded-xl border border-border/80 bg-card/40 p-4 sm:p-5">
        <div className="flex items-center justify-between border-b border-border/60 pb-2.5">
          <p className="eyebrow text-muted-foreground">Session Temperament</p>
          <span className="text-xs text-muted-foreground tabular-nums">{total} total</span>
        </div>
        <div className="mt-4 space-y-2.5">
          {Object.entries(distribution)
            .sort((a, b) => b[1] - a[1])
            .map(([emotion, count]) => (
              <div key={emotion} className="flex items-center gap-3">
                <span className="eyebrow w-20 shrink-0 text-muted-foreground">
                  {EMOTION_META[emotion as keyof typeof EMOTION_META].label}
                </span>
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
                  <div
                    className="h-full rounded-full transition-all duration-500"
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
