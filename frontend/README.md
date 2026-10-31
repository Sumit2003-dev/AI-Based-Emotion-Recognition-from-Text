# Six Signal Sentiment Compass

An intuitive AI system that detects emotions from textual data such as tweets, messages, customer feedback, and reviews. The system classifies text across six foundational emotion signals: **Happy**, **Sad**, **Angry**, **Surprise**, **Fear**, and **Neutral**.

---

## ✨ Features

- **Single Text Emotion Reader**: Paste any tweet, review, or message to see the emotion breakdown and dominant feeling.
- **Batch Analyzer**: Process up to 40 inputs simultaneously with aggregate emotion distributions.
- **Recent Readings History**: Keep track of inspected texts, dominant emotion badges, and score percentages.
- **Zero-Dependency Fallback**: Built-in calibrated NLP rule engine works immediately out of the box with zero external configuration, plus optional integration with local FastAPI or OpenAI models.
- **Modern UI**: Designed with React 19, TanStack Start & Router, Tailwind CSS v4, and Radix UI.

---

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+ recommended)
- npm or pnpm

### Installation

```bash
cd frontend
npm install
```

### Running Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) (or the port indicated in your console) to view the application.

---

## 🧪 Running Tests

To run the test suite:

```bash
npm test
```

---

## 🛠️ Configuration (Optional)

- **OpenAI (Optional)**: Set `OPENAI_API_KEY` in your environment to use an OpenAI model for inference.
- **Local FastAPI Backend (Optional)**: If you are running the `AI-Based Emotion Recognition from Text` FastAPI backend on `http://127.0.0.1:8000`, the app will automatically route predictions to it.
