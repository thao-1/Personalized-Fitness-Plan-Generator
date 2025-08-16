import os
import sys
import requests
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# If BACKEND_URL is not set, run in local mode (call Python modules directly)
BACKEND_URL = os.getenv("BACKEND_URL", "").strip()
LOCAL_MODE = BACKEND_URL == ""

if LOCAL_MODE:
    # Ensure project root is on sys.path when running on Streamlit Cloud
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Load secrets into environment for SDKs that read os.environ
    try:
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"]).strip()
        if "INDEX_DIR" in st.secrets:
            os.environ["INDEX_DIR"] = str(st.secrets["INDEX_DIR"]).strip()
        if "OPENAI_MODEL" in st.secrets:
            os.environ["OPENAI_MODEL"] = str(st.secrets["OPENAI_MODEL"]).strip()
        if "EMBEDDING_MODEL" in st.secrets:
            os.environ["EMBEDDING_MODEL"] = str(st.secrets["EMBEDDING_MODEL"]).strip()
    except Exception:
        pass
    # Lazy import to avoid errors if modules are missing during remote mode
    from backend.planner import Planner
    from backend.models import PlanRequest
    from backend.rag import FitnessRAG
    if "planner" not in st.session_state:
        st.session_state.planner = Planner()
    if "rag" not in st.session_state:
        st.session_state.rag = FitnessRAG()

st.set_page_config(page_title="Personalized Fitness Plan Generator", page_icon="💪", layout="wide")

st.title("💪 Personalized Fitness Plan Generator")
st.caption("Generate a weekly training plan tailored to your goals, experience, and equipment.")

with st.sidebar:
    st.header("Configuration")
    mode_label = "Local (in-app)" if LOCAL_MODE else f"Remote: {BACKEND_URL}"
    st.write(f"Mode: {mode_label}")
    if LOCAL_MODE:
        if st.button("Health Check"):
            st.success("Local mode OK: running planner and RAG in-process")
    else:
        if st.button("Health Check"):
            try:
                r = requests.get(f"{BACKEND_URL}/health", timeout=5)
                st.success(f"API OK: {r.json()}")
            except Exception as e:
                st.error(f"API not reachable: {e}")

st.subheader("Tell us about you")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=13, max_value=100, value=25)
    sex = st.selectbox("Sex (optional)", ["", "male", "female", "other"], index=0)
with col2:
    goal = st.selectbox("Primary Goal", ["fat_loss", "muscle_gain", "strength", "endurance", "general_fitness"], index=1)
    experience = st.selectbox("Experience", ["beginner", "intermediate", "advanced"], index=0)
with col3:
    days = st.slider("Days per week", min_value=1, max_value=7, value=4)
    time_min = st.slider("Time per workout (min)", min_value=15, max_value=180, value=60, step=5)

# Injuries input (comma-separated free text)
injuries_text = st.text_input(
    "Injuries or sensitive areas (optional, comma-separated)",
    placeholder="e.g., shoulder, knee, lower back",
)
injuries = [s.strip() for s in injuries_text.split(",") if s.strip()]

st.subheader("Equipment Available")
default_eq = ["dumbbell", "barbell", "machine", "cable", "bodyweight", "kettlebell", "bands"]
selected_eq = st.multiselect("Select all that apply", default_eq, default=["bodyweight", "dumbbell"])

if st.button("Generate Plan", type="primary"):
    payload = {
        "age": int(age),
        "sex": sex if sex else None,
        "goal": goal,
        "experience": experience,
        "injuries": injuries or [],
        "days_per_week": int(days),
        "time_per_workout_min": int(time_min),
        "equipment_available": selected_eq,
    }
    try:
        with st.spinner("Generating your plan..."):
            if LOCAL_MODE:
                req_model = PlanRequest(**payload)
                plan = st.session_state.planner.generate_plan(req_model)
                # Pydantic model -> dict
                data = plan.dict() if hasattr(plan, "dict") else plan
            else:
                resp = requests.post(f"{BACKEND_URL}/recommend-plan", json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
    except requests.HTTPError as e:
        st.error(f"Server error: {e.response.text}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to generate plan: {e}")
        st.stop()

    st.success("Plan generated!")
    st.markdown(f"**Summary:** {data.get('summary', '')}")

    schedule: List[Dict[str, Any]] = data.get("weekly_schedule", [])
    # Quick rollup to show something even if individual days have no exercises
    total_ex = sum(len(d.get("exercises", [])) for d in schedule)
    st.caption(f"Plan contains {total_ex} exercises across {len(schedule)} day(s).")

    # Debug view of raw API response
    with st.expander("Debug: Raw API response"):
        import json as _json
        st.json(data)
    for day in schedule:
        with st.expander(f"{day.get('day')} — {day.get('focus')}"):
            exercises = day.get("exercises", [])
            if not exercises:
                st.info("No exercises found for this day. Try adjusting equipment, goal, or experience.")
            else:
                for ex in exercises:
                    title = ex.get("title")
                    meta = [
                        ex.get("type"),
                        ex.get("mechanics"),
                        ex.get("level"),
                        ex.get("equipment"),
                    ]
                    meta_str = " · ".join([m for m in meta if m])
                    st.markdown(f"**{title}**" + (f"  \\\n_{meta_str}_" if meta_str else ""))
                    pm = ex.get("primary_muscles") or []
                    sm = ex.get("secondary_muscles") or []
                    if pm or sm:
                        st.caption("Primary: " + ", ".join(pm) + (" | Secondary: " + ", ".join(sm) if sm else ""))
                    if ex.get("notes"):
                        st.write(ex.get("notes"))

    # Simple download as JSON
    import json
    st.download_button(
        "Download Plan (JSON)",
        data=json.dumps(data, indent=2),
        file_name="fitness_plan.json",
        mime="application/json",
    )

st.markdown("---")
st.caption("Tip: Adjust days/week and equipment to see different plan structures.")

# =============================
# RAG Q&A Section
# =============================
st.header("🧠 Exercise Knowledge Q&A (RAG)")
st.write("Build a knowledge index from the dataset and ask questions about exercises, safety, and form.")

col_a, col_b = st.columns([1, 3])
with col_a:
    if st.button("Build/Refresh Index", help="Creates the FAISS index using OpenAI embeddings"):
        try:
            with st.spinner("Building index... this may take up to a minute on first run"):
                if LOCAL_MODE:
                    idx = st.session_state.rag.build_or_refresh_index()
                    st.success(f"Index ready: {idx}")
                else:
                    r = requests.post(f"{BACKEND_URL}/rag/build-index", timeout=120)
                    r.raise_for_status()
                    st.success(f"Index ready: {r.json().get('index_dir')}")
        except requests.HTTPError as e:
            st.error(f"Server error: {e.response.text}")
        except Exception as e:
            st.error(f"Failed to build index: {e}")
with col_b:
    question = st.text_input("Ask a question about exercises (e.g., 'What are safe alternatives for squats if I have knee pain?')")
    k = st.slider("Retriever k", 2, 10, 6)
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Thinking..."):
                    if LOCAL_MODE:
                        ans = st.session_state.rag.qa(question, k=k)
                        st.write(ans)
                    else:
                        r = requests.post(f"{BACKEND_URL}/rag/ask", json={"question": question, "k": k}, timeout=60)
                        r.raise_for_status()
                        st.write(r.json().get("answer", ""))
            except requests.HTTPError as e:
                st.error(f"Server error: {e.response.text}")
            except Exception as e:
                st.error(f"RAG request failed: {e}")
