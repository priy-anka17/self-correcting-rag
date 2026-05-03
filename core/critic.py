"""Critic agent — evaluates answers for hallucinations and factual accuracy."""

from dataclasses import dataclass, field
from typing import List
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from config import LLM_MODEL, get_groq_api_key, GROQ_BASE_URL


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checking AI critic. Your job is to evaluate
whether an AI-generated answer is faithful to the source context.

You must check for:
1. **Hallucinations** — Claims in the answer NOT supported by the context.
2. **Contradictions** — Claims that directly contradict the context.
3. **Missing key info** — Important information in the context that the answer ignores.
4. **Source misattribution** — Incorrect citation of sources.

Return your evaluation as a JSON object with these exact fields:
{{
    "verdict": "PASS" | "FAIL" | "PARTIAL",
    "confidence": 0.0 to 1.0,
    "hallucinations": ["list of hallucinated claims, if any"],
    "contradictions": ["list of contradictions, if any"],
    "missing_info": ["list of important missing information, if any"],
    "reasoning": "Brief explanation of your evaluation",
    "suggested_requery": "If FAIL or PARTIAL, suggest a better search query to get accurate info. Otherwise null"
}}

Be strict. If in doubt, flag it."""),
    ("human", """Source Context:
{context}

Question: {question}

Generated Answer:
{answer}

Evaluate the answer's faithfulness to the source context. Return ONLY valid JSON."""),
])


@dataclass
class CriticResult:
    """Container for critic evaluation results."""

    verdict: Verdict
    confidence: float
    hallucinations: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    reasoning: str = ""
    suggested_requery: str = ""
    raw_response: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS

    @property
    def issue_count(self) -> int:
        return len(self.hallucinations) + len(self.contradictions)

    @property
    def summary(self) -> str:
        parts = [f"Verdict: {self.verdict.value} (confidence: {self.confidence:.0%})"]
        if self.hallucinations:
            parts.append(f"Hallucinations: {', '.join(self.hallucinations)}")
        if self.contradictions:
            parts.append(f"Contradictions: {', '.join(self.contradictions)}")
        if self.missing_info:
            parts.append(f"Missing info: {', '.join(self.missing_info)}")
        parts.append(f"Reasoning: {self.reasoning}")
        return "\n".join(parts)


def critique_answer(question: str, context: str, answer: str) -> CriticResult:
    """Evaluate an answer against its source context for faithfulness."""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        api_key=get_groq_api_key(),
        base_url=GROQ_BASE_URL,
    )
    parser = JsonOutputParser()
    chain = CRITIC_PROMPT | llm | parser

    try:
        result = chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
        })

        return CriticResult(
            verdict=Verdict(result.get("verdict", "FAIL")),
            confidence=float(result.get("confidence", 0.0)),
            hallucinations=result.get("hallucinations", []),
            contradictions=result.get("contradictions", []),
            missing_info=result.get("missing_info", []),
            reasoning=result.get("reasoning", ""),
            suggested_requery=result.get("suggested_requery") or "",
            raw_response=result,
        )
    except Exception as e:
        return CriticResult(
            verdict=Verdict.FAIL,
            confidence=0.0,
            reasoning=f"Critic evaluation failed: {str(e)}",
        )
