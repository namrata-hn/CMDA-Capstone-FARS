from flask import Flask, request
from flask.json import jsonify
from flask_cors import CORS
import logging
import pandas as pd
from orchestration import answer_question

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok", "backend": "databricks-sql"}, 200

    @app.route("/query", methods=["POST"])
    def query():
        try:
            payload = request.get_json(force=True)
            question = payload.get("query") or payload.get("question")

            if not question:
                return jsonify({"error": "Missing 'query' field"}), 400

            logger.info(f"Received question: {question}")

            # NEW: run hybrid SQL+RAG orchestration
            result = answer_question(question)

            # Handle different output shapes
            # If SQL-only or BOTH, `ask_fars_database()` returns a dict
            if isinstance(result, dict):
                sql_query = result.get("query", "")
                df = result.get("results", pd.DataFrame())
                nl_answer = result.get("answer", "")

                # Convert dataframe to JSON
                json_output = {
                    "query": sql_query,
                    "columns": list(df.columns),
                    "rows": df.fillna("").astype(str).values.tolist(),
                    "answer": nl_answer,
                    "row_count": len(df),
                }
                return jsonify(json_output), 200

            # If RAG-only or a text explanation, it's a plain string
            if isinstance(result, str):
                return jsonify({
                    "query": None,
                    "columns": [],
                    "rows": [],
                    "answer": result,
                    "row_count": 0
                }), 200

            # Fallback
            return jsonify({
                "error": "Unexpected result type",
                "details": str(type(result))
            }), 500

        except Exception as e:
            logger.exception("Unexpected server error")
            return jsonify({
                "error": "Internal server error",
                "details": str(e)
            }), 500

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    # Disable reloader to avoid double LLM initialization
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)