"""
StudPerf FastAPI Backend
========================
Exposes three endpoints consumed by dashboard.html:

  POST /predict  — Run the real ML models (LR + RF) on a student profile
  POST /chat     — Stream a Gemini/Gemma coaching response via SSE
  GET  /health   — Liveness probe

Run with:
    uvicorn api:app --reload --port 8000
"""

import asyncio
import json
import os
import pickle
import queue
import threading
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv("project.env", override=True)

app = FastAPI(title="StudPerf API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry (loaded once at startup) ────────────────────
META: dict = {}


@app.on_event("startup")
def startup() -> None:
    path = "student_performance_models.pkl"
    if not os.path.exists(path):
        raise RuntimeError(
            f"Model file not found: {path}. Run `python run_analysis.py` first."
        )
    with open(path, "rb") as f:
        META.update(pickle.load(f))
    print("✓ ML models loaded from", path)


# ── Request / Response schemas ─────────────────────────────────

class StudentProfile(BaseModel):
    hours: int = 18
    attendance: int = 88
    prev_scores: int = 75
    tutoring: int = 2
    physical: int = 3
    sleep: int = 7
    parental: str = "Medium"
    motivation: str = "Medium"
    resources: str = "Medium"
    family_income: str = "Medium"
    teacher_quality: str = "Medium"
    peer_influence: str = "Neutral"
    internet: str = "Yes"
    extracurricular: str = "No"
    disability: str = "No"
    parent_edu: str = "College"
    distance: str = "Near"
    school_type: str = "Private"
    gender: str = "Female"


class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    profile: StudentProfile
    history: List[ChatMessage] = []
    predicted_score: Optional[float] = None


# ── Preprocessing helper ───────────────────────────────────────

def _profile_to_df(p: StudentProfile) -> pd.DataFrame:
    return pd.DataFrame([{
        "Hours_Studied":          p.hours,
        "Attendance":             p.attendance,
        "Parental_Involvement":   p.parental,
        "Access_to_Resources":    p.resources,
        "Extracurricular_Activities": p.extracurricular,
        "Sleep_Hours":            p.sleep,
        "Previous_Scores":        p.prev_scores,
        "Motivation_Level":       p.motivation,
        "Internet_Access":        p.internet,
        "Tutoring_Sessions":      p.tutoring,
        "Family_Income":          p.family_income,
        "Teacher_Quality":        p.teacher_quality,
        "School_Type":            p.school_type,
        "Peer_Influence":         p.peer_influence,
        "Physical_Activity":      p.physical,
        "Learning_Disabilities":  p.disability,
        "Parental_Education_Level": p.parent_edu,
        "Distance_from_Home":     p.distance,
        "Gender":                 p.gender,
    }])


# ── /predict ───────────────────────────────────────────────────

@app.post("/predict")
def predict(profile: StudentProfile) -> dict:
    if not META:
        raise HTTPException(503, "Models not loaded yet.")

    df = _profile_to_df(profile)
    cat_cols = META["cat_features_for_model"]

    # Linear Regression model uses drop_first=True encoding
    df_stats = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_stats = df_stats.reindex(columns=META["X_encoded_stats_columns"], fill_value=0)

    # Random Forest classifier uses drop_first=False encoding
    df_clf = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    df_clf = df_clf.reindex(columns=META["X_encoded_columns"], fill_value=0)

    score    = float(META["lr_stats_model"].predict(df_stats)[0])
    pass_prob = float(META["clf_model"].predict_proba(df_clf)[0][1] * 100)

    risk = "low" if score >= 70 else ("medium" if score >= 65 else "high")

    return {
        "score":     round(score, 1),
        "pass_prob": round(pass_prob, 1),
        "risk":      risk,
    }


# ── Gemini API key reader ──────────────────────────────────────

def _get_api_key() -> str:
    """Read the key from project.env first, fall back to env var."""
    if os.path.exists("project.env"):
        with open("project.env") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "GEMINI_API_KEY":
                        return v.strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY", "")


# ── System instruction builder ─────────────────────────────────

def _system_instruction(profile: StudentProfile, score: float) -> str:
    return f"""You are Aura, an elite Academic Success Coach for secondary school students.

Student Profile (use this data in every response):
  Predicted Exam Score : {score:.1f} / 100
  Weekly Study Hours   : {profile.hours} hrs
  Attendance Rate      : {profile.attendance}%
  Previous Scores      : {profile.prev_scores}%
  Tutoring / Month     : {profile.tutoring} sessions
  Sleep / Night        : {profile.sleep} hrs
  Physical Activity    : {profile.physical} days/week
  Motivation Level     : {profile.motivation}
  Parental Involvement : {profile.parental}
  Peer Influence       : {profile.peer_influence}
  Access to Resources  : {profile.resources}
  Internet Access      : {profile.internet}
  Family Income        : {profile.family_income}
  Teacher Quality      : {profile.teacher_quality}
  Extracurricular      : {profile.extracurricular}
  School Type          : {profile.school_type}
  Distance from School : {profile.distance}
  Gender               : {profile.gender}

Coaching rules:
- Be motivating, candid, and practical.
- Use simple English. No jargon. No emojis.
- Keep responses concise and structured with clear headers.
- Every suggestion must directly reference the student's specific data points above.
- Do not mention machine learning, regression, probability, or analytics to the student.
- Focus only on actionable academic improvement steps.
- When asked to generate a full coaching report, use this structure:
  1. Performance Snapshot (2-3 lines)
  2. Key Strengths (2 bullet points)
  3. Main Areas to Improve (2-3 bullet points)
  4. 4-Week Improvement Plan (Week 1-4 with Focus + Action)
  5. 3 Atomic Daily Habits
  6. SMART Goal
  7. Final Coach Message (2-3 motivating lines)"""


# ── SSE streaming generator ────────────────────────────────────

async def _sse_stream(prompt: str, sys_instr: str, api_key: str):
    """
    Bridge the synchronous google-genai streaming generator into
    an async FastAPI StreamingResponse via a thread + queue.
    """
    q: queue.Queue = queue.Queue()

    def _worker():
        client = genai.Client(api_key=api_key)
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        model_sequence = ["gemma-4-26b-a4b-it", "gemini-2.5-flash"]

        for model in model_sequence:
            try:
                cfg = types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
                )
                stream = client.models.generate_content_stream(
                    model=model, contents=contents, config=cfg
                )
                for chunk in stream:
                    if chunk.text:
                        q.put(("text", chunk.text))
                q.put(("done", None))
                return
            except Exception as exc:
                if model == model_sequence[-1]:
                    q.put(("error", str(exc)))
                    q.put(("done", None))
                # else: silently try the next model

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        kind, val = await loop.run_in_executor(None, q.get)
        if kind == "done":
            yield "data: [DONE]\n\n"
            break
        elif kind == "error":
            yield f"data: {json.dumps({'error': val})}\n\n"
            yield "data: [DONE]\n\n"
            break
        else:
            yield f"data: {json.dumps({'text': val})}\n\n"


# ── /chat ──────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    api_key = _get_api_key()
    if len(api_key) < 10:
        raise HTTPException(
            503,
            "Gemini API key not configured. "
            "Add GEMINI_API_KEY=<your-key> to project.env and restart the server.",
        )

    score = req.predicted_score or 67.0
    sys_instr = _system_instruction(req.profile, score)

    # Weave conversation history into the prompt
    history_lines = []
    for msg in req.history[-10:]:
        speaker = "Student" if msg.role == "user" else "Aura"
        history_lines.append(f"{speaker}: {msg.content}")

    if history_lines:
        prompt = "\n\n".join(history_lines) + f"\n\nStudent: {req.message}"
    else:
        prompt = req.message

    return StreamingResponse(
        _sse_stream(prompt, sys_instr, api_key),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── /health ────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": bool(META)}
