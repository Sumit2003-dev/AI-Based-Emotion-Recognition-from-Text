# 🧠 AI-Based Emotion Recognition Frontend

> **Created & Developed by [Sumit Kumar](https://github.com/Sumit2003-dev)**

An intuitive AI web application that detects emotions from textual data such as tweets, messages, customer feedback, and reviews. Powered by React 19, TanStack Start & Router, and Tailwind CSS v4, connecting seamlessly to the Python FastAPI NLP & Hybrid Neural Transformer backend.

---

## ✨ Features

- **Single Text Emotion Reader**: Paste any tweet, review, or message to see the emotion breakdown, confidence score, and psychological guidance.
- **Batch Analyzer**: Process bulk inputs simultaneously with aggregate emotion distributions and CSV export.
- **Recent Readings History**: Keep track of inspected texts, dominant emotion badges, and score percentages.
- **Dual-Engine Support**: Automatically connects to the local FastAPI backend (Hybrid DistilRoBERTa + LinearSVC) with local fallback.
- **Modern UI**: Designed with React 19, TanStack Start, and Tailwind CSS v4.

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
