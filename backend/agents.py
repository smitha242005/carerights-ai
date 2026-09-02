"""
agents.py — The three core agents (+ second-opinion + debate + safety gate) for CareRights AI.

Every agent is grounded in retrieved source chunks (see rag.py). Agents are explicitly
instructed to say "not covered by available sources" rather than guess when retrieval is
weak — this is the actual mechanism that keeps the system honest, not just a prompt line.

Uses Google Gemini's free API tier (no credit card required) via Google AI Studio.
Get a free key at https://aistudio.google.com/apikey and set it as GEMINI_API_KEY.
"""

import os
import json
from google import genai
from google.genai import types
from rag import retrieve_medical, retrieve_insurance

MODEL = "gemini-3.6-flash"  # current stable free-tier model (gemini-2.5-flash was retired)

_client = None

def _get_client():
    """Lazy client creation — so the rest of the app (signup, login, RAG, etc.) still
    works even before GEMINI_API_KEY is set. The error only appears when an agent
    actually tries to run, not when this module is merely imported."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Get a free key (no credit card needed) at "
                "https://aistudio.google.com/apikey and set it with: "
                "export GEMINI_API_KEY=your_key_here"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _call_claude(system_prompt: str, user_content: str, max_tokens: int = 2048) -> str:
    """Calls Gemini. Kept this function name (_call_claude) so nothing else in this
    file needs renaming — it's just the "call the LLM" function now, provider-agnostic."""
    response = _get_client().models.generate_content(
        model=MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),  # Gemini 3.x uses thinking_level (not thinking_budget, which is 2.5-only) — MINIMAL keeps the token budget for the actual answer
        ),
    )
    return response.text


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"The model's response wasn't valid JSON (this usually means the reply got "
            f"cut off — try raising max_tokens). Raw response was: {cleaned[:300]}..."
        ) from e


def _format_sources(chunks: list[dict]) -> str:
    if not chunks:
        return "(No sufficiently relevant sources were retrieved for this query.)"
    lines = []
    for c in chunks:
        lines.append(f"[Source {c['chunk_id']}] {c['topic']}: {c['content']}")
    return "\n\n".join(lines)


AGENT_INSTRUCTIONS = """You are the {role} agent inside CareRights AI, a healthcare
decision-support system.

You will be given a user's case description AND a set of retrieved source excerpts.

RULES:
- Base your finding ONLY on the retrieved sources provided below. Do not use outside
  knowledge to fill gaps.
- If the retrieved sources do not adequately cover the user's situation, say so explicitly
  in your reasoning and set confidence to "gray-area" — do NOT guess or invent a citation.
- Always mention which source chunk_id(s) support your finding, if any.
- Never give a definitive diagnosis or legal ruling — frame findings as informational.
- Write the "finding" and "reasoning" fields in {language}. Keep "confidence" and
  "cited_sources" values in English (they are structured codes, not prose).

Respond ONLY with strict JSON, no preamble, no markdown fences, in exactly this shape:
{{"finding": "one short sentence verdict", "reasoning": "2-3 sentence explanation grounded in the sources", "confidence": "strong" | "moderate" | "gray-area", "cited_sources": ["CHUNK-ID", ...]}}

{focus}
"""

MEDICAL_FOCUS = "Focus only on whether the treatment/test described is typically considered medically necessary, based on the retrieved medical guideline excerpts."
RIGHTS_FOCUS = "Focus only on whether insurance would typically be expected to cover this, what timelines apply, and what rights/escalation options exist, based on the retrieved insurance/regulatory excerpts."
RISK_FOCUS = "Focus only on what could realistically happen if this is delayed or denied, and how urgent it is, based on the retrieved medical guideline excerpts about delay risk."


def run_medical_agent(case_text: str, language: str = "English") -> dict:
    sources = retrieve_medical(case_text, top_k=3)
    system = AGENT_INSTRUCTIONS.format(role="Medical-Evidence", focus=MEDICAL_FOCUS, language=language)
    user = f"USER CASE:\n{case_text}\n\nRETRIEVED SOURCES:\n{_format_sources(sources)}"
    result = _parse_json(_call_claude(system, user))
    result["retrieved_sources"] = sources
    return result


def run_rights_agent(case_text: str, language: str = "English") -> dict:
    sources = retrieve_insurance(case_text, top_k=3)
    system = AGENT_INSTRUCTIONS.format(role="Cost/Insurance-Rights", focus=RIGHTS_FOCUS, language=language)
    user = f"USER CASE:\n{case_text}\n\nRETRIEVED SOURCES:\n{_format_sources(sources)}"
    result = _parse_json(_call_claude(system, user))
    result["retrieved_sources"] = sources
    return result


def run_risk_agent(case_text: str, language: str = "English") -> dict:
    # Risk agent reuses the medical corpus (delay-risk chunks live there)
    sources = retrieve_medical(case_text, top_k=3)
    system = AGENT_INSTRUCTIONS.format(role="Risk-of-Waiting", focus=RISK_FOCUS, language=language)
    user = f"USER CASE:\n{case_text}\n\nRETRIEVED SOURCES:\n{_format_sources(sources)}"
    result = _parse_json(_call_claude(system, user))
    result["retrieved_sources"] = sources
    return result


SECOND_OPINION_SYSTEM = """You are the Second-Opinion agent inside CareRights AI. Your job
is to challenge the Medical-Evidence agent's finding — actively look for reasons it might be
wrong or incomplete, the way a real second medical opinion would. You are given the same
retrieved sources the first agent used.

Write the "challenge" field in {language}.

Respond ONLY with strict JSON, no preamble, no markdown fences:
{{"agrees_with_medical_agent": true | false, "challenge": "1-3 sentences on what might be missing, wrong, or worth double-checking — or why you agree if you do"}}
"""


def run_second_opinion_agent(case_text: str, medical_result: dict, language: str = "English") -> dict:
    sources_text = _format_sources(medical_result.get("retrieved_sources", []))
    user = (f"USER CASE:\n{case_text}\n\n"
            f"MEDICAL AGENT'S FINDING:\n{json.dumps({k: v for k, v in medical_result.items() if k != 'retrieved_sources'})}\n\n"
            f"SOURCES THE MEDICAL AGENT USED:\n{sources_text}")
    system = SECOND_OPINION_SYSTEM.format(language=language)
    return _parse_json(_call_claude(system, user))


DEBATE_SYSTEM = """You compare three agent outputs (Medical, Insurance-Rights,
Risk-of-Waiting) plus a second-opinion challenge, for a healthcare decision-support system.

Write the "summary" field in {language}.

Respond ONLY with strict JSON, no preamble, no markdown fences:
{{"agreement": "agree" | "conflict", "summary": "3-4 sentence plain-language explanation of where the agents align or clash and why it matters for the user", "high_risk": true | false}}

Set "high_risk" true ONLY if the case describes something acutely life-threatening or
emergency-level (e.g. severe chest pain, stroke symptoms, active bleeding), where the user
should be told to seek immediate real medical/legal help instead of reading agent output.
"""


def run_debate(case_text: str, medical: dict, rights: dict, risk: dict, second_opinion: dict, language: str = "English") -> dict:
    payload = {
        "case": case_text,
        "medical": {k: v for k, v in medical.items() if k != "retrieved_sources"},
        "rights": {k: v for k, v in rights.items() if k != "retrieved_sources"},
        "risk": {k: v for k, v in risk.items() if k != "retrieved_sources"},
        "second_opinion": second_opinion,
    }
    system = DEBATE_SYSTEM.format(language=language)
    return _parse_json(_call_claude(system, json.dumps(payload)))


APPEAL_LETTER_SYSTEM = """You write formal insurance appeal letters for CareRights AI users.
You are given: the user's case, the Medical-Evidence agent's finding (with its cited source),
and the Cost/Insurance-Rights agent's finding (with its cited source).

Write a complete, ready-to-send formal appeal letter that:
- Is addressed generically to "The Grievance Redressal Officer" (the user will fill in the
  insurer's name and their own details)
- States clearly what is being appealed
- Cites the specific medical necessity finding, referencing which guideline supports it
- Cites the specific insurance/rights finding, referencing the relevant regulation or
  escalation right
- Requests a specific next action (reconsideration of the claim) and references the
  standard escalation path if the appeal itself is not resolved
- Ends with a polite, professional closing
- Is written in {language}

Do NOT invent any citation, guideline, or regulation not present in the findings given to
you. If either finding is marked "gray-area" or has no strong citation, phrase that part of
the letter more generally (e.g. request clarification) rather than asserting something not
supported by the source.

Respond with the letter text only — no JSON, no markdown formatting, no preamble like
"Here is the letter." Just the letter itself, ready to copy or print.
"""


def generate_appeal_letter(case_text: str, medical: dict, rights: dict, language: str = "English") -> str:
    """Generates a ready-to-send appeal letter grounded in the same findings already
    shown to the user — no new retrieval, just turns the existing grounded findings into
    a usable document. Called on-demand (not part of the main pipeline) to save API quota
    for users who don't need it."""
    payload = {
        "case": case_text,
        "medical_finding": {k: v for k, v in medical.items() if k != "retrieved_sources"},
        "rights_finding": {k: v for k, v in rights.items() if k != "retrieved_sources"},
    }
    system = APPEAL_LETTER_SYSTEM.format(language=language)
    return _call_claude(system, json.dumps(payload), max_tokens=1500).strip()


def analyze_case(case_text: str, language: str = "English") -> dict:
    """Full pipeline: 3 agents -> second opinion -> debate/safety synthesis."""
    medical = run_medical_agent(case_text, language=language)
    rights = run_rights_agent(case_text, language=language)
    risk = run_risk_agent(case_text, language=language)
    second_opinion = run_second_opinion_agent(case_text, medical, language=language)
    debate = run_debate(case_text, medical, rights, risk, second_opinion, language=language)
    return {
        "medical": medical,
        "rights": rights,
        "risk": risk,
        "second_opinion": second_opinion,
        "debate": debate,
    }


if __name__ == "__main__":
    # Quick manual test — requires GEMINI_API_KEY to be set. Run: python agents.py
    test_case = "My insurance company denied coverage for a knee replacement my doctor recommended after six months of physiotherapy did not help."
    result = analyze_case(test_case)
    print(json.dumps(result, indent=2))
