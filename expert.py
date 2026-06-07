from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from legacy_mind.llm import call_llm
from legacy_mind.retrieval import Chunk, knowledge_map, retrieve


@dataclass
class ExpertDNA:
    name: str
    expertise: list[str]
    decision_style: str
    risk_tolerance: str
    communication_style: str
    evidence_summary: str


def generate_expert_dna(expert_name: str, chunks: list[Chunk]) -> ExpertDNA:
    context = _context(chunks[:8])
    prompt = f"""
Analyze the expert knowledge below and produce compact JSON with these keys:
name, expertise, decision_style, risk_tolerance, communication_style, evidence_summary.

Expert name: {expert_name}

Knowledge:
{context}
"""
    response = call_llm(
        "You extract an expert profile from documents. Return only valid JSON.",
        prompt,
    )
    if response:
        parsed = _parse_json(response)
        if parsed:
            return ExpertDNA(
                name=str(parsed.get("name") or expert_name or "Expert"),
                expertise=_as_list(parsed.get("expertise")),
                decision_style=str(parsed.get("decision_style") or "Evidence-led"),
                risk_tolerance=str(parsed.get("risk_tolerance") or "Medium"),
                communication_style=str(parsed.get("communication_style") or "Clear and structured"),
                evidence_summary=str(parsed.get("evidence_summary") or "Profile inferred from uploaded documents."),
            )

    topics = [term.title() for term, _ in knowledge_map(chunks, limit=5)]
    return ExpertDNA(
        name=expert_name or "Expert",
        expertise=topics or ["Domain Expertise"],
        decision_style="Evidence-led and practical",
        risk_tolerance="Medium",
        communication_style="Structured, detailed, and recommendation-focused",
        evidence_summary="Offline profile inferred from the most repeated concepts in the uploaded knowledge base.",
    )


def answer_as_expert(question: str, dna: ExpertDNA, chunks: list[Chunk]) -> str:
    relevant = retrieve(question, chunks, k=5)
    context = _context(relevant)
    prompt = f"""
Expert DNA:
{json.dumps(asdict(dna), indent=2)}

Retrieved knowledge:
{context}

Question:
{question}

Answer as the expert. Ground the response in the retrieved knowledge, state assumptions, and end with next steps.
"""
    response = call_llm("You are a document-grounded expert assistant.", prompt)
    if response:
        return response

    evidence = relevant[0].text if relevant else "No uploaded evidence was available."
    return (
        f"As {dna.name}, I would approach this using a {dna.decision_style.lower()} style. "
        f"The most relevant uploaded evidence says: \"{evidence[:420]}\". "
        "My recommendation is to verify facts first, identify the highest-risk unknowns, compare options against business impact, "
        "and document the decision trail. Next steps: collect missing evidence, define risk owners, and choose the lowest-regret action."
    )


def _context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{chunk.source} #{chunk.index + 1}] {chunk.text}" for chunk in chunks)


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _as_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return ["Domain Expertise"]
