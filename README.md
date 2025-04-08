
# 🌾 Agrosarthi Backend

Agrosarthi is an AI-powered agricultural assistant tailored for farmers in Maharashtra. This FastAPI backend provides real-time predictions using machine learning models for crop selection, price estimation, and yield forecasting.

> ✅ **Deployed using Docker on Google Cloud Run for scalable, serverless infrastructure.**

---

## 🚀 Features

- 🌱 **Crop Prediction** – Recommends the most suitable crop based on soil and weather conditions.
- 📈 **Price Estimation** – Predicts crop prices using market, location, and seasonal factors.
- 🌾 **Yield Prediction** – Estimates expected crop yield based on area, crop type, and region.
- 💬 **AI Chatbot (Coming Soon)** – Integrates with Google Gemini API to answer agriculture-related queries.

---

## 🛠️ Tech Stack

- **FastAPI** – Lightweight, fast Python web framework
- **Joblib** – Model loading for ML predictions
- **Pandas & NumPy** – Data processing
- **Pydantic** – Data validation for API requests
- **CORS Middleware** – Cross-origin resource sharing
- **Google Cloud Storage** – Public hosting of `.pkl` ML model files
- **Docker** – Containerized application
- **Google Cloud Run** – Serverless deployment of backend API

---

## 📁 Folder Structure

```
agrosarthi-backend/
├── app.py                       # Main FastAPI app with all API routes
├── model/
│   ├── Crop_prediction.pkl      # Crop prediction model
│   ├── crop_price_model.pkl     # Crop price prediction model
│   └── yield_prediction_model.pkl # Yield prediction model
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker image definition
└── README.md                    # You're here!
```

---

## 📦 API Endpoints

### ✅ `/predict-crop`
**Method**: POST  
**Description**: Predicts the most suitable crop.  
**Payload**:
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 23.0,
  "humidity": 80.5,
  "ph": 6.5,
  "rainfall": 200.0
}
```
**Response**:
```json
{
  "predicted_crop": "rice"
}
```

---

### ✅ `/predict-price`
**Method**: POST  
**Description**: Estimates market price for a given crop and location.  
**Payload**:
```json
{
  "crop": "soybean",
  "district": "Pune",
  "market": "Pune",
  "month": "October",
  "season": "Kharif",
  "agri_season": "Monsoon"
}
```
**Response**:
```json
{
  "predicted_price": 3250.75
}
```

---

### ✅ `/predict-yield`
**Method**: POST  
**Description**: Estimates yield in kg/hectare for a crop and district.  
**Payload**:
```json
{
  "state": "Maharashtra",
  "district": "Nashik",
  "crop": "wheat",
  "area": 2.5,
  "season": "Rabi"
}
```
**Response**:
```json
{
  "predicted_yield": 2175.40
}
```

---

## ☁️ Deployment Done on: Docker + Google Cloud Run

### ✅ Step 1: Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

### ✅ Step 2: Build & Push Docker Image

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/agrosarthi-backend
```

---

### ✅ Step 3: Deploy to Cloud Run

```bash
gcloud run deploy agrosarthi-backend \
  --image gcr.io/YOUR_PROJECT_ID/agrosarthi-backend \
  --platform managed \
  --region asia-south1 \
  --allow-unauthenticated
```

You’ll receive a public Cloud Run URL (e.g. `https://agrosarthi-backend-abc123.a.run.app`).

---

## 🔗 Live API

🌐 [Agrosarthi Backend API](https://agrosarthi-backend-885337506715.asia-south1.run.app)  
🧠 [Agrosarthi Frontend ](https://agrosarthi-frontend.web.app/)  
📂 [GitHub Repo](https://github.com/your-username/agrosarthi-backend)

---

## 📌 Future Roadmap

- 🔤 Multilingual support
- 📉 Weather-integrated real-time yield forecasts
- 📊 Farmer dashboard for visualization
- 🧠 Full AI chatbot integration
- 📱 Mobile version for offline support

---

## 👨‍🔬 Maintainers

Built with My Team AlgoCraft For GDG campus Solution Submission
- Piyush Bhavsar (Me)
- Arjan Pathan
- Tanishqa Jagtap
