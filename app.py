from flask import Flask, request, jsonify
import joblib
import pefile

app = Flask(__name__)

# Load the trained model
model = joblib.load('malware_model.pkl')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Process the file for malware detection
    result = detect_malware(file)
    
    return jsonify({'result': result})

def detect_malware(file):
    # Extract features from the file using pefile
    pe = pefile.PE(file)
    features = extract_features(pe)
    
    # Predict using the loaded model
    prediction = model.predict([features])
    
    return 'malicious' if prediction[0] == 1 else 'safe'

def extract_features(pe):
    # Implement feature extraction logic here
    return []

if __name__ == '__main__':
    app.run(debug=True)
