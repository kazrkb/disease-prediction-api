# Disease Prediction API

A Flask API that predicts diseases based on symptoms using a trained KNN model.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict disease from symptoms |
| `/symptoms` | GET | Get all available symptoms |
| `/health` | GET | Health check |

## Usage

### Predict Disease
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["fever", "headache", "fatigue"]}'
```

### Get All Symptoms
```bash
curl https://your-app.onrender.com/symptoms
```

## Local Development
```bash
pip install -r requirements.txt
python app.py
```
