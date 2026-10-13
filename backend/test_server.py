"""Test-only entrypoint: runs the real app with agents.analyze_case mocked,
so we can verify the frontend renders real backend responses correctly
without needing a live Gemini API key."""
from unittest.mock import patch

FAKE_RESULT = {
    "medical": {"finding": "Likely medically necessary", "reasoning": "Conservative treatment (physiotherapy) was tried for six months first, consistent with standard practice before surgical referral.", "confidence": "strong", "cited_sources": ["MED-001", "MED-007"], "retrieved_sources": [{"chunk_id": "MED-001", "topic": "Knee replacement: conservative treatment first", "content": "Total knee replacement is generally considered after at least three to six months of documented conservative treatment (physiotherapy, medication, weight management) has failed to relieve pain and restore function.", "source_file": "medical_guidelines.md", "relevance_score": 0.412}, {"chunk_id": "MED-007", "topic": "Documentation of failed treatment", "content": "Records of physiotherapy sessions and imaging support a necessity claim.", "source_file": "medical_guidelines.md", "relevance_score": 0.233}]},
    "rights": {"finding": "Denial is challengeable through the standard appeal process", "reasoning": "A denial reason of 'not medically necessary' alone, without supporting clinical rationale, can be formally appealed via the Grievance Redressal Officer.", "confidence": "moderate", "cited_sources": ["INS-008", "INS-004"], "retrieved_sources": [{"chunk_id": "INS-008", "topic": "Appealing a denial", "content": "A denial can be appealed to the insurer's Grievance Redressal Officer, and then to the Insurance Ombudsman if unresolved.", "source_file": "insurance_rights.md", "relevance_score": 0.318}, {"chunk_id": "INS-004", "topic": "Reasons for rejection", "content": "Insurers must state the reason for rejection in writing.", "source_file": "insurance_rights.md", "relevance_score": 0.201}]},
    "risk": {"finding": "Moderate risk if delayed further", "reasoning": "Delaying joint replacement can lead to further cartilage loss and reduced surgical outcomes, though this is a functional risk rather than an emergency.", "confidence": "moderate", "cited_sources": ["MED-005"], "retrieved_sources": [{"chunk_id": "MED-005", "topic": "Delay and joint deterioration", "content": "Prolonged delay in joint replacement can lead to further cartilage loss.", "source_file": "medical_guidelines.md", "relevance_score": 0.27}]},
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

agents.generate_next_steps = lambda *a, **k: {
    "steps": [
        {"title": "Ask for the denial reason in writing", "detail": "Request the insurer's written reason for rejection (INS-004)."},
        {"title": "Collect your physiotherapy records", "detail": "Six months of documented treatment supports necessity (MED-001)."},
        {"title": "File an appeal with the Grievance Redressal Officer", "detail": "Use the appeal letter below, then escalate to the Ombudsman if unresolved (INS-008)."},
    ],
    "note": "[TEST MODE] This is a fake plan so you can check the layout.",
}


def _fake_edit(letter, instruction, case_text, medical, rights, language="English"):
    """Pretend editor so you can try the chat box without spending Gemini quota."""
    text = instruction.lower()
    if "short" in text:
        new = letter.split("\n\n")[0] + "\n\n[TEST MODE] Shortened version.\n\nSincerely,\n[Your name]"
    else:
        new = letter + "\n\n[TEST MODE edit applied: " + instruction + "]"
    return {"reply": "(test mode) I applied your change: " + instruction, "letter": new}

main_mod_agents = agents
main_mod_agents.edit_appeal_letter = _fake_edit

import uvicorn
import main as main_module
main_module.edit_appeal_letter = _fake_edit
main_module.generate_next_steps = agents.generate_next_steps

if __name__ == "__main__":
    uvicorn.run(main_module.app, host="127.0.0.1", port=8000)