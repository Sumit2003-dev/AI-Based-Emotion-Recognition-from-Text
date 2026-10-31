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
- **FastAPI Backend (Optional)**: Set `FASTAPI_URL` to route predictions to your deployed Python backend (e.g. on Render, Railway, or Hugging Face Spaces). Defaults to `http://127.0.0.1:8000` in local development, and seamlessly falls back to the built-in local inference engine if no external backend is reachable.

---

## 🌐 Deploying to Vercel

The application is pre-configured with Nitro's Vercel Build Output API v3 adapter for fast serverless and edge deployment on [Vercel](https://vercel.com).

### Deployment Steps:

1. **Import your GitHub repository**:
   - Go to [vercel.com/new](https://vercel.com/new).
   - Select your repository: `Sumit2003-dev/AI-Based-Emotion-Recognition-from-Text`.

2. **Configure Root Directory**:
   - Under **Build and Output Settings**, find **Root Directory**.
   - Click **Edit** and set it to `frontend` ⚠️ _(Crucial for monorepo detection)_.

3. **Verify Build Settings**:
   - **Framework Preset**: Auto-detected (Vite / TanStack Start).
   - **Build Command**: `npm run build`
   - **Output Directory**: `.vercel/output` (managed automatically by Nitro).

4. **Environment Variables (Optional)**:
   - `FASTAPI_URL`: Your hosted FastAPI backend endpoint (if hosting Python separately on Render/Railway).
   - `OPENAI_API_KEY`: (Optional) For alternate GPT-4o-mini classification.

5. **Deploy**:
   - Click **Deploy**. Vercel will install dependencies, build the static client assets, package serverless functions, and assign your live production URL.
