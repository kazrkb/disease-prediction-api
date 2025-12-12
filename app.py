from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

import os

app = Flask(__name__)
CORS(app)  # Enable CORS for PHP requests

# Load the trained model and symptom list
model = joblib.load('disease_model.pkl')
symptom_list = joblib.load('symptom_list.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease based on symptoms.
    Expects JSON: {"symptoms": ["fever", "headache", "fatigue"]}
    """
    try:
        data = request.get_json()
        user_symptoms = data.get('symptoms', [])
        
        if not user_symptoms:
            return jsonify({
                'success': False,
                'error': 'No symptoms provided'
            }), 400
        
        # Create feature vector (all zeros initially)
        features = np.zeros(len(symptom_list))
        
        matched_symptoms = []
        unmatched_symptoms = []
        
        # Set 1 for symptoms user has
        for symptom in user_symptoms:
            # Clean symptom name to match dataset format
            symptom_clean = symptom.lower().strip().replace(' ', '_')
            
            if symptom_clean in symptom_list:
                idx = symptom_list.index(symptom_clean)
                features[idx] = 1
                matched_symptoms.append(symptom_clean)
            else:
                unmatched_symptoms.append(symptom)
        
        if not matched_symptoms:
            return jsonify({
                'success': False,
                'error': 'No valid symptoms found. Please check symptom names.',
                'available_symptoms': symptom_list[:20]  # Show first 20 as example
            }), 400
        
        # Predict disease
        prediction = model.predict([features])[0]
        
        # Get prediction probability if available
        try:
            probabilities = model.predict_proba([features])[0]
            confidence = float(max(probabilities) * 100)
        except:
            confidence = None
        
        return jsonify({
            'success': True,
            'disease': prediction,
            'confidence': confidence,
            'matched_symptoms': matched_symptoms,
            'unmatched_symptoms': unmatched_symptoms,
            'message': f"Based on your symptoms, you may have: {prediction}. Please consult a doctor for proper diagnosis."
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """Return list of all available symptoms"""
    return jsonify({
        'success': True,
        'symptoms': symptom_list,
        'total': len(symptom_list)
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'symptoms_count': len(symptom_list)
    })

if __name__ == '__main__':
    print("Disease Prediction API is running...")
    print(f"Loaded {len(symptom_list)} symptoms")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
