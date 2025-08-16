# Personalized Fitness Plan Generator

URL: https://thao-1-personalized-fitness-plan-generator.streamlit.app/

An AI-driven app that generates a weekly workout plan tailored to a user's goals, experience, available equipment, and constraints, using a simple rule-based expert system over the `megaGymDataset.csv`.

- Front-end: Streamlit
- Backend API: FastAPI (`/recommend-plan`)
- Data: `megaGymDataset.csv` at project root

## Quick Start

1) Create and activate a virtual environment (recommended)

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run the backend API

```
uvicorn backend.main:app --reload --port 8000
```

4) In a new terminal, run the Streamlit app

```
export BACKEND_URL=http://127.0.0.1:8000
streamlit run frontend/app.py
```

Then open the Streamlit URL shown in the terminal (usually http://localhost:8501).

## RAG (Retrieval-Augmented Generation)

This project includes an optional RAG module to answer free-form questions about exercises (form, safety, substitutions) using your dataset + OpenAI.

What this does (and doesn't):
- It does not train a new model. It builds a FAISS vector index over text derived from `megaGymDataset.csv` and uses OpenAI for embeddings and answering.
- You need an `OPENAI_API_KEY` set in environment variables (or `.env`).

### Configure environment

Copy `.env.example` to `.env` and set your key:

```
cp .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini            # optional
EMBEDDING_MODEL=text-embedding-3-small  # optional
INDEX_DIR=data/index
BACKEND_URL=http://127.0.0.1:8000
```

### Build the FAISS index

You can build from Streamlit by clicking "Build/Refresh Index" in the RAG section, or via API:

```
curl -X POST http://127.0.0.1:8000/rag/build-index
```

### Ask questions

From Streamlit, type your question and click "Ask", or via API:

```
curl -X POST http://127.0.0.1:8000/rag/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are knee-friendly squat alternatives?", "k": 6}'
```

## Endpoints

- `GET /health` — health check
- `POST /recommend-plan` — generate plan

Example payload:
```json
{
  "age": 28,
  "sex": "male",
  "goal": "muscle_gain",
  "experience": "beginner",
  "injuries": ["shoulder"],
  "days_per_week": 4,
  "time_per_workout_min": 60,
  "equipment_available": ["bodyweight", "dumbbell", "machine"]
}
```

## Architecture (Block Diagram)

```
User → Streamlit UI
           │
           ▼
     FastAPI Backend (/recommend-plan)
           │
           ▼
   Planner (Rule-based + Filters)
   ├─ Load megaGymDataset.csv
   ├─ Filter by equipment / experience / injuries
   ├─ Build weekly split by goal + days/week
   └─ Rank & select exercises per day
           │
           ▼
   JSON Plan → Streamlit renders schedule

            ┌──────────────────────────────────────────────────────────┐
            │                        RAG Module                        │
            │  Build FAISS index from CSV (OpenAI embeddings)         │
            │  Streamlit → FastAPI /rag/ask → RetrievalQA (OpenAI LLM)│
            └──────────────────────────────────────────────────────────┘
```

## Notes
- The planner uses deterministic sampling for demo reproducibility.
- Columns in `megaGymDataset.csv` are normalized heuristically; common aliases are handled. If your CSV has different column names, adjust `backend/planner.py` → `Exercise.from_row` to map them.
- You can enhance with LLM/RAG explanations later by adding another FastAPI endpoint and a small vector index over exercise descriptions.

## Project Structure
```
.
├── backend
│   ├── __init__.py
│   ├── main.py          # FastAPI app & endpoints
│   ├── models.py        # Pydantic request/response schemas
│   └── planner.py       # Rule-based expert system over CSV
├── frontend
│   └── app.py           # Streamlit UI
├── megaGymDataset.csv   # Dataset (provided)
├── requirements.txt
└── README.md
```

## Video & Report Guidance
- Show a live demo generating plans for multiple profiles.
- In the video, explain: request schema, planner rules, filtering logic, and how the weekly split is computed.
- Include the block diagram above in your short report and describe the problem and importance (personalized, constraint-aware workouts).
