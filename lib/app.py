
from __future__ import annotations

from flask import Flask, jsonify, request

from lib.ai_client import generate_response
from lib.company_documents import COMPANY_DOCUMENTS
from lib.rag_service import build_prompt, retrieve_context, source_metadata


def create_app():
    app = Flask(__name__)

    @app.get("/api/health")
    def health_check():
        return jsonify({"status": "ok"})

    @app.post("/api/ask")
    def ask_question():
        # Read JSON safely
        data = request.get_json(silent=True)

        # Validate the request body
        if not isinstance(data, dict):
            return jsonify({
                "error": "A valid query is required."
            }), 400

        query = data.get("query")

        if not isinstance(query, str) or not query.strip():
            return jsonify({
                "error": "Query must be a non-empty string."
            }), 400

        query = query.strip()

        # Retrieve relevant company documents
        context_matches = retrieve_context(
            query,
            COMPANY_DOCUMENTS
        )

        # Handle questions without relevant context
        if not context_matches:
            return jsonify({
                "query": query,
                "answer": (
                    "The available documents do not contain "
                    "enough information to answer this question."
                ),
                "sources": []
            }), 200

        # Build the prompt
        prompt = build_prompt(query, context_matches)

        # Call the AI model and handle failures
        try:
            answer = generate_response(prompt)
        except RuntimeError:
            return jsonify({
                "error": "The AI model service is currently unavailable."
            }), 503

        # Prepare source metadata
        sources = [
            source_metadata(match)
            for match in context_matches
        ]

        # Return the completed response
        return jsonify({
            "query": query,
            "answer": answer,
            "sources": sources
        }), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
