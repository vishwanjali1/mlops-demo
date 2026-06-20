from flask import Flask, jsonify, request
import joblib
import os
from pathlib import Path

app = Flask(__name__)
MODEL_PATH = Path("artifacts/model.pkl")

if not MODEL_PATH.exists():
    # train if model does not exist
    import train as _train
    _train.main()

model = joblib.load(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400
    
    try:
        prediction = model.predict([data['features']])
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)