from flask import Flask, request, jsonify
from flask_cors import CORS
from router import answer_question

app = Flask(__name__)
CORS(app)   # allow React to call Flask

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        response = answer_question(question)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)