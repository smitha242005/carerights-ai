"""
main.py — CareRights AI backend API.

Run with:  uvicorn main:app --reload --port 8000
Requires:  export GEMINI_API_KEY=your_key_here
Optional:  export DATABASE_URL=postgresql://...   (keeps accounts after Render restarts)
"""

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
import io

import database as db
from agents import analyze_case, generate_appeal_letter, edit_appeal_letter, generate_next_steps, check_providers

app = FastAPI(title="CareRights AI API")

# In local development, allow any origin so it's easy to test.
# In production, set ALLOWED_ORIGINS to your real frontend URL(s), comma-separated,
# e.g. ALLOWED_ORIGINS=https://carerights-ai.netlify.app
_allowed = os.environ.get("ALLOWED_ORIGINS")
allow_origins = [o.strip() for o in _allowed.split(",")] if _allowed else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    db.init_db()


# ---------- case-type classifier (real, deterministic, no AI call needed) ----------

CASE_TYPE_KEYWORDS = {
    "Insurance Denial": ["denied", "denial", "reject", "rejected", "declined", "not covered", "won't cover", "wont cover", "refused"],
    "Treatment Necessity": ["necessary", "necessity", "recommend", "recommended", "need surgery", "need a", "should i get", "doctor wants", "doctor says i need"],
    "Delay / Prior Authorization": ["prior authorization", "pre-authorization", "preauthorization", "waiting for approval", "delay", "delayed", "taking too long", "still waiting"],
    "Urgent / Emergency": ["chest pain", "emergency", "severe pain", "can't breathe", "cant breathe", "bleeding", "urgent", "immediately"],
}


def classify_case_type(case_text: str) -> str:
    text_lower = case_text.lower()
    scores = {}
    for case_type, keywords in CASE_TYPE_KEYWORDS.items():
        scores[case_type] = sum(1 for kw in keywords if kw in text_lower)
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "General Inquiry"
    return best_type


# ---------- request/response models ----------

class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class CaseRequest(BaseModel):
    case_text: str
    language: str = "English"  # English, Hindi, Tamil, Telugu, Malayalam


class LetterChatRequest(BaseModel):
    letter: str       # the letter exactly as it currently appears on screen
    message: str      # what the user wants changed, e.g. "make it shorter"


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class FeedbackRequest(BaseModel):
    case_id: int
    agent: str  # "medical" | "rights" | "risk"
    rating: str  # "up" | "down"


# ---------- auth helper ----------

def get_current_user(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    user = db.get_user_from_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    return user


# ---------- auth endpoints ----------

@app.post("/api/signup")
def signup(req: SignupRequest):
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    try:
        user = db.create_user(req.name, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/api/login")
def login(req: LoginRequest):
    user = db.verify_user(req.email, req.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    token = db.create_session(user["id"])
    return {"token": token, "user": user}


@app.post("/api/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        db.delete_session(authorization.removeprefix("Bearer ").strip())
    return {"ok": True}


# ---------- case endpoints ----------

@app.post("/api/cases")
def create_case(req: CaseRequest, authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    if not req.case_text.strip():
        raise HTTPException(status_code=400, detail="Case description cannot be empty.")
    case_type = classify_case_type(req.case_text)
    try:
        result = analyze_case(req.case_text, language=req.language)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent pipeline failed: {e}")
    result["case_type"] = case_type
    result["language"] = req.language
    saved = db.save_case(user["id"], req.case_text, result)
    return saved


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    case = db.get_case_by_id(req.case_id, user["id"])
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    if req.agent not in ("medical", "rights", "risk"):
        raise HTTPException(status_code=400, detail="agent must be 'medical', 'rights', or 'risk'.")
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'.")
    db.save_feedback(user["id"], req.case_id, req.agent, req.rating)
    return {"ok": True}


@app.get("/api/cases")
def list_cases(authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    return db.get_cases_for_user(user["id"])


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    case = db.get_case_by_id(case_id, user["id"])
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return case


@app.post("/api/cases/{case_id}/appeal-letter")
def create_appeal_letter(case_id: int, authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    case = db.get_case_by_id(case_id, user["id"])
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    result = case["result"]
    if "medical" not in result or "rights" not in result:
        raise HTTPException(status_code=400, detail="This case has no medical/rights findings to base a letter on (it may have been safety-blocked).")
    try:
        letter = generate_appeal_letter(
            case["case_text"], result["medical"], result["rights"],
            language=result.get("language", "English"),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Letter generation failed: {e}")
    return {"letter": letter}


@app.post("/api/cases/{case_id}/appeal-letter/chat")
def chat_edit_appeal_letter(case_id: int, req: LetterChatRequest, authorization: str | None = Header(default=None)):
    """Conversational editing of the appeal letter. Only called when the user sends a
    chat message, so it uses Gemini quota only on request."""
    user = get_current_user(authorization)
    case = db.get_case_by_id(case_id, user["id"])
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    result = case["result"]
    if "medical" not in result or "rights" not in result:
        raise HTTPException(status_code=400, detail="This case has no findings to base a letter on.")
    message = req.message.strip()
    letter = req.letter.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Please type what you want changed.")
    if len(message) > 1000:
        raise HTTPException(status_code=400, detail="Message is too long (max 1000 characters).")
    if not letter or len(letter) > 10000:
        raise HTTPException(status_code=400, detail="The letter is empty or too long to edit.")
    try:
        edited = edit_appeal_letter(
            letter, message, case["case_text"], result["medical"], result["rights"],
            language=result.get("language", "English"),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Letter editing failed: {e}")
    return edited


@app.post("/api/cases/{case_id}/next-steps")
def create_next_steps(case_id: int, authorization: str | None = Header(default=None)):
    """Grounded 3-5 step action plan. On request only, to save API quota."""
    user = get_current_user(authorization)
    case = db.get_case_by_id(case_id, user["id"])
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    result = case["result"]
    if result.get("debate", {}).get("high_risk"):
        raise HTTPException(status_code=400, detail="This case was safety-blocked. Please seek real medical help first.")
    if not all(k in result for k in ("medical", "rights", "risk")):
        raise HTTPException(status_code=400, detail="This case has no findings to base a plan on.")
    try:
        return generate_next_steps(case["case_text"], result["medical"], result["rights"], result["risk"],
                                   language=result.get("language", "English"))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not build the plan: {e}")


MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 10
MAX_TEXT_CHARS = 4000


@app.post("/api/extract-text")
async def extract_text(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    """Reads a denial letter (PDF or .txt) and returns its text so the case box can be filled.
    The file is processed in memory and is NOT stored."""
    get_current_user(authorization)
    name = (file.filename or "").lower()
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (max 5 MB).")
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")

    pages = 0
    if name.endswith(".pdf"):
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="This does not look like a real PDF file.")
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise HTTPException(status_code=400, detail="This PDF is password-protected.")
            pages = len(reader.pages)
            text = "\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PDF_PAGES])
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read this PDF.")
    elif name.endswith(".txt"):
        text = data.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(status_code=400, detail="Please upload a PDF or .txt file.")

    text = " ".join(text.split())  # collapse whitespace / line breaks
    if len(text) < 20:
        raise HTTPException(status_code=422, detail="No readable text found. Scanned or photo PDFs are not supported — please type or paste the key details instead.")
    truncated = len(text) > MAX_TEXT_CHARS
    return {"text": text[:MAX_TEXT_CHARS], "truncated": truncated, "pages": pages}


@app.get("/api/me")
def get_me(authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    return db.get_profile(user["id"])


@app.post("/api/me/change-password")
def change_password(req: ChangePasswordRequest, authorization: str | None = Header(default=None)):
    user = get_current_user(authorization)
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    token = authorization.removeprefix("Bearer ").strip()
    if not db.change_password(user["id"], req.current_password, req.new_password, keep_token=token):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    return {"ok": True}


@app.get("/api/llm-check")
def llm_check():
    """Diagnostic: tests Groq and Gemini with a 1-word request and reports timing/errors.
    Open this URL in a browser if the app feels slow. Never shows your keys."""
    return check_providers()


@app.get("/api/health")
def health():
    # "database" tells you whether accounts are kept permanently (postgresql) or in a
    # file that Render's free plan erases on restart (sqlite).
    return {"status": "ok", "database": db.backend_name()}