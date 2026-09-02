"""Test-only entrypoint: runs the real app with agents.analyze_case mocked,
so we can verify the frontend renders real backend responses correctly
without needing a live Gemini API key."""
from unittest.mock import patch

FAKE_RESULT = {
    "medical": {"finding": "Likely medically necessary", "reasoning": "Conservative treatment (physiotherapy) was tried for six months first, consistent with standard practice before surgical referral.", "confidence": "strong", "cited_sources": ["MED-001", "MED-007"]},
    "rights": {"finding": "Denial is challengeable through the standard appeal process", "reasoning": "A denial reason of 'not medically necessary' alone, without supporting clinical rationale, can be formally appealed via the Grievance Redressal Officer.", "confidence": "moderate", "cited_sources": ["INS-008", "INS-004"]},
    "risk": {"finding": "Moderate risk if delayed further", "reasoning": "Delaying joint replacement can lead to further cartilage loss and reduced surgical outcomes, though this is a functional risk rather than an emergency.", "confidence": "moderate", "cited_sources": ["MED-005"]},
    "second_opinion": {"agrees_with_medical_agent": True, "challenge": "Agrees with the finding — six months of documented failed physiotherapy is a well-established threshold for surgical referral."},
    "debate": {"agreement": "conflict", "summary": "The Medical and Risk agents both support proceeding with surgery, and the Rights agent confirms the denial can be appealed rather than accepted as final. The conflict here is with the insurer's decision, not between the agents themselves — all three agree the case supports moving forward.", "high_risk": False},
}

import agents
agents.analyze_case = lambda case_text, language="English": FAKE_RESULT
agents.generate_appeal_letter = lambda case_text, medical, rights, language="English": (
    "To: The Grievance Redressal Officer\n\n"
    "Subject: Appeal of Claim Denial — [TEST MODE, no real Gemini call made]\n\n"
    "This is a fake placeholder letter returned by test_server.py so you can verify the "
    "frontend button, layout, and PDF export work correctly before spending real API quota.\n\n"
    "Sincerely,\n[Your name]"
)

import uvicorn
import main as main_module

if __name__ == "__main__":
    uvicorn.run(main_module.app, host="127.0.0.1", port=8000)
