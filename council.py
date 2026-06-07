from __future__ import annotations

from dataclasses import dataclass

from legacy_mind.expert import ExpertDNA
from legacy_mind.llm import call_llm
from legacy_mind.retrieval import Chunk, retrieve


@dataclass
class CouncilOpinion:
    role: str
    stance: str
    reasoning: str
    recommendation: str


@dataclass
class CouncilResult:
    opinions: list[CouncilOpinion]
    final_recommendation: str


COUNCIL_ROLES = {
    "Cyber Expert": "Focus on technical containment, evidence preservation, attacker behavior, and operational security.",
    "Finance Expert": "Focus on cost, cash risk, insurance, regulatory exposure, and financial downside.",
    "Operations Expert": "Focus on service continuity, customer impact, recovery timeline, and organizational execution.",
}


def run_council(question: str, dna: ExpertDNA, chunks: list[Chunk]) -> CouncilResult:
    relevant = retrieve(question, chunks, k=6)
    context = "\n\n".join(chunk.text for chunk in relevant)
    opinions = [_role_opinion(role, lens, question, dna, context) for role, lens in COUNCIL_ROLES.items()]
    final = _final_summary(question, opinions)
    return CouncilResult(opinions=opinions, final_recommendation=final)


def _role_opinion(role: str, lens: str, question: str, dna: ExpertDNA, context: str) -> CouncilOpinion:
    prompt = f"""
Question: {question}
Primary expert: {dna.name}
Expert DNA: {dna}
Retrieved knowledge: {context}

Respond as {role}. Lens: {lens}
Use this exact format:
Stance: ...
Reasoning: ...
Recommendation: ...
"""
    response = call_llm(f"You are the {role} in an expert decision council.", prompt)
    if response:
        return _parse_opinion(role, response)

    stance = "Cautious proceed"
    if role == "Cyber Expert":
        stance = "Do not decide until containment and evidence are verified"
        recommendation = "Isolate affected systems, preserve logs, identify blast radius, and validate recovery options."
    elif role == "Finance Expert":
        stance = "Compare payment risk against recovery and legal exposure"
        recommendation = "Model downtime cost, insurance coverage, sanctions risk, and negotiation downside before approving spend."
    else:
        stance = "Protect continuity while recovery work proceeds"
        recommendation = "Prioritize critical services, communicate clearly, and run a staged restoration plan."

    return CouncilOpinion(
        role=role,
        stance=stance,
        reasoning=f"{lens} The uploaded knowledge suggests using {dna.name}'s {dna.decision_style.lower()} decision style.",
        recommendation=recommendation,
    )


def _final_summary(question: str, opinions: list[CouncilOpinion]) -> str:
    opinion_text = "\n".join(
        f"{op.role}: {op.stance} | {op.reasoning} | {op.recommendation}" for op in opinions
    )
    prompt = f"""
Question: {question}
Council opinions:
{opinion_text}

Create a concise final recommendation with: decision, rationale, immediate actions, and risks to monitor.
"""
    response = call_llm("You are the final judge agent for an expert council.", prompt)
    if response:
        return response

    return (
        "Final recommendation: choose the lowest-regret path after validating facts. "
        "Do not make a high-stakes decision from pressure alone. Contain the incident, quantify business impact, check legal and insurance constraints, "
        "then decide with executive approval. Immediate actions: preserve evidence, stabilize critical operations, brief leadership, and document assumptions. "
        "Risks to monitor: hidden compromise, regulatory exposure, reputational damage, and recovery delays."
    )


def _parse_opinion(role: str, text: str) -> CouncilOpinion:
    fields = {"stance": "", "reasoning": "", "recommendation": ""}
    current = None
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("stance:"):
            current = "stance"
            fields[current] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("reasoning:"):
            current = "reasoning"
            fields[current] = stripped.split(":", 1)[1].strip()
        elif lower.startswith("recommendation:"):
            current = "recommendation"
            fields[current] = stripped.split(":", 1)[1].strip()
        elif current and stripped:
            fields[current] += " " + stripped

    return CouncilOpinion(
        role=role,
        stance=fields["stance"] or "Conditional",
        reasoning=fields["reasoning"] or text,
        recommendation=fields["recommendation"] or "Review the evidence and proceed with controlled execution.",
    )
