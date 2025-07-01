# backend.py
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

conversation_history = []

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    history_string = "\n".join(conversation_history)
    inputs = tokenizer.encode_plus(history_string, user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_history.append(user_input)
    conversation_history.append(response)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
