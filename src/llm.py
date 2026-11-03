import os
from typing import List

from dotenv import load_dotenv
from groq import Groq


load_dotenv()  # reads GROQ_API_KEY from .env in project root

MODEL = "openai/gpt-oss-20b"   # small, fast, free on Groq
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file in the project root."
            )
        _client = Groq(api_key=api_key)
    return _client


def _chat(prompt: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def summarize_chunks(chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = (
        "You are given excerpts from a single PDF document. "
        "Write a concise summary (5-8 sentences) of what the document is about, "
        "using only the information in the excerpts.\n\n"
        f"EXCERPTS:\n{context}"
    )
    return _chat(prompt)


def answer_question(question: str, chunks: List[str]) -> str:
    context = "\n\n".join(chunks)
    prompt = (
        "Answer the user's question using ONLY the context below. "
        "If the context does not contain the answer, say so honestly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )
    return _chat(prompt)