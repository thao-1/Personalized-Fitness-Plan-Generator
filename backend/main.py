from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PlanRequest, PlanResponse
from .planner import Planner
from pydantic import BaseModel
from .rag import FitnessRAG, RAGConfig

app = FastAPI(title="Personalized Fitness Plan API", version="0.1.0")

# CORS (allow Streamlit on localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

planner = Planner()
rag = FitnessRAG()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend-plan", response_model=PlanResponse)
def recommend_plan(request: PlanRequest):
    try:
        plan = planner.generate_plan(request)
        return plan
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate plan: {e}")


# --- RAG endpoints ---
class AskBody(BaseModel):
    question: str
    k: int = 6


@app.post("/rag/build-index")
def rag_build_index():
    try:
        path = rag.build_or_refresh_index()
        return {"status": "ok", "index_dir": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build index: {e}")


@app.post("/rag/ask")
def rag_ask(body: AskBody):
    try:
        answer = rag.qa(body.question, k=body.k)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")
