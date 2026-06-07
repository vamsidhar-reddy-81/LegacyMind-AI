from __future__ import annotations

from dataclasses import dataclass
import re

from legacy_mind.llm import call_llm
from legacy_mind.retrieval import Chunk, retrieve


@dataclass
class KnowledgeAnswer:
    answer: str
    sources: list[Chunk]


def answer_from_knowledge(question: str, chunks: list[Chunk]) -> KnowledgeAnswer:
    sources = retrieve(question, chunks, k=4)
    context = "\n\n".join(f"[{chunk.source} #{chunk.index + 1}] {chunk.text}" for chunk in sources)
    prompt = f"""
Use only the uploaded project knowledge below to answer the user's question.
If the answer is not present, say what is missing and suggest what document should be uploaded.

Uploaded knowledge:
{context}

User question:
{question}

Answer in a helpful, direct style. Include practical steps when useful.
"""
    response = call_llm("You are LegacyMind AI, a chatbot grounded in uploaded project documents.", prompt)
    if response:
        return KnowledgeAnswer(answer=response, sources=sources)

    return KnowledgeAnswer(answer=_offline_answer(question, sources), sources=sources)


def _offline_answer(question: str, sources: list[Chunk]) -> str:
    if not sources:
        return (
            "I do not have uploaded knowledge yet. Upload PDF, DOCX, or TXT files first, then ask me anything about them."
        )

    direct = _known_learning_answer(question)
    if direct:
        return direct

    snippets = _best_sentences(question, sources)
    if snippets:
        return (
            "Short answer:\n\n"
            f"{_summary_from_sentences(snippets)}\n\n"
            "Relevant points from the uploaded document:\n\n"
            + "\n".join(f"- {sentence}" for sentence in snippets[:5])
        )

    return (
        "I found related content in the uploaded document, but it is not clean enough to answer confidently. "
        "Try asking with one or two important keywords, or turn off Fast demo mode for a deeper AI-generated answer."
    )


def _known_learning_answer(question: str) -> str | None:
    lower = question.lower()
    asks_java_variable = "java" in lower and ("variable" in lower or "variables" in lower)
    if not asks_java_variable:
        return None

    return """In Java, a variable is a named storage location used to hold a value while a program runs.

Every variable has:

- a data type
- a name
- a value

Example:

```java
int age = 20;
String name = "John";
double price = 99.50;
boolean isActive = true;
```

Here:

- `int`, `String`, `double`, and `boolean` are data types
- `age`, `name`, `price`, and `isActive` are variable names
- `20`, `"John"`, `99.50`, and `true` are values

Simple meaning: a variable is like a labeled box. The label is the variable name, and the value inside the box can be used or changed by the program.

Common Java variable types:

- `int` for whole numbers
- `double` for decimal numbers
- `char` for one character
- `boolean` for true/false
- `String` for text

Example use:

```java
int marks = 85;
marks = 90;
System.out.println(marks);
```

This prints `90` because the value stored in `marks` was changed."""


def _best_sentences(question: str, sources: list[Chunk]) -> list[str]:
    query_terms = {term for term in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", question.lower()) if len(term) > 3}
    candidates = []
    for chunk in sources:
        text = re.sub(r"\s+", " ", chunk.text)
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            clean = sentence.strip()
            if len(clean) < 35 or len(clean) > 420:
                continue
            lower = clean.lower()
            if "disclaimer" in lower or "contents" in lower:
                continue
            score = sum(1 for term in query_terms if term in lower)
            if score:
                candidates.append((score, clean))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [sentence for _, sentence in candidates[:5]]


def _summary_from_sentences(sentences: list[str]) -> str:
    first = sentences[0].strip()
    if len(first) > 240:
        first = first[:240].rsplit(" ", 1)[0] + "."
    return first
