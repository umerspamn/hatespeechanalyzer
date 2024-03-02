from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model_file.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Assuming JSON data is sent from the frontend
    text_data = data["text"]  # Extract the text data from the request
    # Preprocess the text data (similar to how you preprocessed it during training)
    # For example: Convert text to lowercase and remove punctuation
    preprocessed_text = text_data.lower().replace('[^\w\s]', '')
    # Make prediction using the loaded model
    prediction = model.predict([preprocessed_text])[0]
    # Return the prediction as JSON response
    return jsonify({"prediction": "Hate Speech" if prediction == 1 else "Not Hate Speech"})

if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask application
