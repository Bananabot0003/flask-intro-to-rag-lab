
from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> set[str]:
    """Convert text into searchable lowercase tokens."""
    words = re.findall(r"[\w']+", text.lower())

    tokens = {
        word.strip("'")
        for word in words
    }

    return {
        token
        for token in tokens
        if len(token) > 1 and token not in STOPWORDS
    }


def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields."""
    title = document.get("title", "")
    category = document.get("category", "")
    tags = document.get("tags", [])
    body = document.get("text", "")

    if isinstance(tags, list):
        tags = " ".join(str(tag) for tag in tags)

    return f"{title} {category} {tags} {body}"


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Calculate relevance using keyword overlap."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(document.get("title", ""))

    matched_terms = query_tokens.intersection(document_tokens)

    score = len(matched_terms)
    score += 0.5 * len(query_tokens.intersection(title_tokens))

    return {
        "document": document,
        "score": score,
        "matched_terms": sorted(matched_terms),
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Retrieve the highest-scoring matching documents."""

    scored_documents = [
        score_document(query, document)
        for document in documents
    ]

    relevant_documents = [
        match
        for match in scored_documents
        if match["score"] >= minimum_score
    ]

    relevant_documents.sort(
        key=lambda match: match["score"],
        reverse=True
    )

    return relevant_documents[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents for the AI prompt."""
    if not context_matches:
        return "No relevant context was found."

    blocks = []

    for match in context_matches:
        document = match["document"]

        block = (
            f"Source ID: {document.get('id', '')}\n"
            f"Title: {document.get('title', '')}\n"
            f"Category: {document.get('category', '')}\n"
            f"Content: {document.get('text', '')}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a grounded prompt for the AI model."""
    context = format_context(context_matches)

    return f"""Instructions:
You are an internal company documentation assistant.
Answer using only the provided context.
Do not invent facts, policies, or procedures.
If the context does not contain the answer, say so.

Context:
{context}

Question:
{query}

Response requirements:
- Provide a clear and accurate answer.
- Use only information supported by the context.
- Do not include unsupported details.
- Refer to relevant sources when appropriate.
"""


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return safe source attribution metadata."""
    document = match["document"]

    return {
        "id": str(document["id"]),
        "title": document["title"],
    }
