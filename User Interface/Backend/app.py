from flask import Flask, request
from flask.json import jsonify
from flask_cors import CORS
import logging
import pandas as pd
from sql_query_chain import ask_fars_database

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
            # Parse request
            payload = request.get_json(force=True)
            question = payload.get("query") or payload.get("question")

            if not question:
                logger.error("Missing query field in request")
                return jsonify({"error": "Missing 'query' field"}), 400

            logger.info(f"Received question: {question}")

            # Run the FARS database query
            result = ask_fars_database(question)
            
            # Log the result structure for debugging
            logger.info(f"Result keys: {result.keys()}")

            # Extract pieces with safer defaults
            sql_query = result.get("query", "")
            df = result.get("results", pd.DataFrame())
            nl_answer = result.get("answer", "")

            # Check if query generation failed
            if sql_query is None or sql_query == "":
                logger.error(f"SQL generation failed: {nl_answer}")
                return jsonify({
                    "error": "Failed to generate SQL query",
                    "details": nl_answer
                }), 500

            # Check if DataFrame is valid
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Invalid DataFrame returned: {type(df)}")
                return jsonify({
                    "error": "Query execution returned invalid data format"
                }), 500

            # Check if query execution had an error
            if "error" in nl_answer.lower() or "exception" in nl_answer.lower():
                logger.error(f"Query execution error: {nl_answer}")
                return jsonify({
                    "error": "Query execution failed",
                    "details": nl_answer,
                    "query": sql_query
                }), 500

            # Convert DataFrame to JSON-serializable format
            try:
                json_output = {
                    "query": sql_query,
                    "columns": list(df.columns),
                    "rows": df.fillna("").astype(str).values.tolist(),
                    "answer": nl_answer,
                    "row_count": len(df)
                }
                
                logger.info(f"Successfully processed query. Returned {len(df)} rows")
                return jsonify(json_output), 200
                
            except Exception as e:
                logger.exception("Error converting DataFrame to JSON")
                return jsonify({
                    "error": "Failed to format results",
                    "details": str(e)
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