"""Answer generator — produces answers from retrieved context."""

from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import LLM_MODEL, get_groq_api_key, GROQ_BASE_URL


GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise, helpful AI research assistant. Answer the user's
question using ONLY the provided context. Follow these rules strictly:

1. Base your answer EXCLUSIVELY on the provided context.
2. If the context doesn't contain enough information, say "Based on the available
   documents, I cannot fully answer this question" and explain what's missing.
3. Cite specific sources using [Source N] references.
4. Be detailed but concise. Use bullet points for clarity when appropriate.
5. Never fabricate information not present in the context."""),
    ("human", """Context:
{context}

Question: {question}

Provide a well-structured answer based solely on the context above."""),
])


@dataclass
class GenerationResult:
    """Container for generation results."""

    answer: str
    question: str
    context_used: str
    model: str


def get_llm(temperature: float = 0.1) -> ChatOpenAI:
    """Create a Groq-backed LLM instance (OpenAI-compatible API)."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=get_groq_api_key(),
        base_url=GROQ_BASE_URL,
    )


def generate_answer(question: str, context: str) -> GenerationResult:
    """Generate an answer from context using the LLM."""
    llm = get_llm()
    chain = GENERATION_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    return GenerationResult(
        answer=response.content,
        question=question,
        context_used=context,
        model=LLM_MODEL,
    )
