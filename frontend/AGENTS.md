# Six Signal Sentiment Compass

## Guidelines for Developers & AI Assistants

- **Purpose**: Six Signal Sentiment Compass evaluates text across 6 primary emotional signals: Happy, Sad, Angry, Surprise, Fear, and Neutral.
- **Architecture**:
  - Frontend: React 19, TanStack Router & Start, Tailwind CSS v4, Lucide icons, Radix UI.
  - Emotion Classifier Engine: `src/lib/emotion/classify.functions.ts` handles scoring via local calibrated sentiment heuristics, optional FastAPI NLP service, or OpenAI.
- **Design Language**: Minimalist typography (Instrument Serif, Work Sans), bespoke color coding per emotion (`--emotion-happy`, `--emotion-sad`, etc.), and clean responsive layout.
