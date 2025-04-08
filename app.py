# ----------------------------- IMPORTS -----------------------------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field  # Add the missing import here
import os
import httpx
import joblib
import pandas as pd
import numpy as np
from fastapi.responses import JSONResponse
import requests
import re
from google import genai

# ----------------------------- FASTAPI APP INITIALIZATION -----------------------------
app = FastAPI()

# ----------------------------- CORS SETUP -----------------------------
# Allow frontend to call the backend without CORS issues

origins = [
    "http://localhost:5500",  # For local testing (default frontend port)
    "http://127.0.0.1:5500",
    "https://agrosarthi-frontend.web.app",  # Deployed frontend URL
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ----------------------------- MODEL LOADING -----------------------------
# Load Crop Prediction Model

crop_model_path = os.path.join('model', 'Crop_prediction.pkl')
try:
    crop_model = joblib.load(crop_model_path)
except Exception as e:
    print(f"Error loading crop prediction model: {e}")
    crop_model = None

# Load Price Estimation Model
price_model_path = "model/crop_price_model.pkl"
try:
    price_model = joblib.load(price_model_path)
except Exception as e:
    print(f"Error loading price estimation model: {e}")
    price_model = None

# Load yeild Prediction Model
yield_model_path = "model/yield_prediction_model.pkl"
try:
    yield_model = joblib.load(yield_model_path)
except Exception as e:
    print(f"Error loading crop prediction model: {e}")
    yeild_model = None

# ----------------------------- INPUT SCHEMAS -----------------------------
# Request schema for Crop Prediction API

class CropPredictionInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=140)
    phosphorus: float = Field(..., ge=0, le=140)
    potassium: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=300)
    temperature: float = Field(..., ge=0, le=50)

# Price Estimation Input Schema
class PriceEstimationRequest(BaseModel):
    district: int
    month: int
    market: int
    commodity: int
    variety: int
    agri_season: int
    climate_season: int

# ----------------------------- STATIC MAPPINGS -----------------------------
# These lists will be used to map human-readable names to index values for model input

districts =['Ahmednagar', 'Akola', 'Amarawati', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Chattrapati Sambhajinagar', 'Dharashiv(Usmanabad)', 'Dhule', 'Gadchiroli', 'Hingoli', 'Jalana', 'Jalgaon', 'Kolhapur', 'Latur', 'Mumbai', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sholapur', 'Thane', 'Vashim', 'Wardha', 'Yavatmal']
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
markets =['ACF Agro Marketing', 'Aarni', 'Aatpadi', 'Achalpur', 'Aheri', 'Ahmednagar', 'Ahmedpur', 'Akhadabalapur', 'Akkalkot', 'Akkalkuwa', 'Akluj', 'Akola', 'Akole', 'Akot', 'Alibagh', 'Amalner', 'Amarawati', 'Ambad (Vadigodri)', 'Ambejaogai', 'Amrawati(Frui & Veg. Market)', 'Anajngaon', 'Armori(Desaiganj)', 'Arvi', 'Ashti', 'Ashti(Jalna)', 'Ashti(Karanja)', 'Aurad Shahajani', 'Ausa', 'BSK Krishi Bazar Private Ltd', 'Babhulgaon', 'Balapur', 'Baramati', 'Barshi', 'Barshi Takli', 'Barshi(Vairag)', 'Basmat', 'Basmat(Kurunda)', 'Beed', 'Bhadrawati', 'Bhagyoday Cotton and Agri Market', 'Bhandara', 'Bhivandi', 'Bhiwapur', 'Bhokar', 'Bhokardan', 'Bhokardan(Pimpalgaon Renu)', 'Bhusaval', 'Bodwad', 'Bori', 'Bori Arab', 'Buldhana', 'Buldhana(Dhad)', 'Chakur', 'Chalisgaon', 'Chandrapur', 'Chandrapur(Ganjwad)', 'Chandur Bazar', 'Chandur Railway', 'Chandvad', 'Chattrapati Sambhajinagar', 'Chikali', 'Chimur', 'Chopada', 'Cottoncity Agro Foods Private Ltd', 'Darwha', 'Daryapur', 'Deglur', 'Deoulgaon Raja', 'Deulgaon Raja Balaji Agro Marketing Private Market', 'Devala', 'Devani', 'Dhadgaon', 'Dhamngaon-Railway', 'Dharangaon', 'Dharashiv', 'Dharmabad', 'Dharni', 'Dhule', 'Digras', 'Dindori', 'Dindori(Vani)', 'Dondaicha', 'Dondaicha(Sindhkheda)', 'Dound', 'Dudhani', 'Fulmbri', 'Gadhinglaj', 'Gajanan Krushi Utpanna Bazar (India) Pvt Ltd', 'Gangakhed', 'Gangapur', 'Gevrai', 'Ghansawangi', 'Ghatanji', 'Ghoti', 'Gondpimpri', 'Gopal Krishna Agro', 'Hadgaon', 'Hadgaon(Tamsa)', 'Hari Har Khajagi Bazar Parisar', 'Higanghat Infrastructure Private Limited', 'Himalyatnagar', 'Hinganghat', 'Hingna', 'Hingoli', 'Hingoli(Kanegoan Naka)', 'Indapur', 'Indapur(Bhigwan)', 'Indapur(Nimgaon Ketki)', 'Islampur', 'J S K Agro Market', 'Jafrabad', 'Jagdamba Agrocare', 'Jai Gajanan Krishi Bazar', 'Jalana', 'Jalgaon', 'Jalgaon Jamod(Aasalgaon)', 'Jalgaon(Masawat)', 'Jalkot', 'Jalna(Badnapur)', 'Jamkhed', 'Jamner', 'Jamner(Neri)', 'Janata Agri Market (DLS Agro Infrastructure Pvt Lt', 'Jawala-Bajar', 'Jawali', 'Jaykissan Krushi Uttpan Khajgi Bazar', 'Jintur', 'Junnar', 'Junnar(Alephata)', 'Junnar(Narayangaon)', 'Junnar(Otur)', 'Kada', 'Kada(Ashti)', 'Kai Madhavrao Pawar Khajgi Krushi Utappan Bazar Sa', 'Kaij', 'Kalamb', 'Kalamb (Dharashiv)', 'Kalamnuri', 'Kalmeshwar', 'Kalvan', 'Kalyan', 'Kamthi', 'Kandhar', 'Kannad', 'Karad', 'Karanja', 'Karjat', 'Karjat(Raigad)', 'Karmala', 'Katol', 'Khamgaon', 'Khed', 'Khed(Chakan)', 'Khultabad', 'Kille Dharur', 'Kinwat', 'Kisan Market Yard', 'Kolhapur', 'Kolhapur(Malkapur)', 'Kopargaon', 'Koregaon', 'Korpana', 'Krushna Krishi Bazar', 'Kurdwadi', 'Kurdwadi(Modnimb)', 'Lakhandur', 'Lasalgaon', 'Lasalgaon(Niphad)', 'Lasalgaon(Vinchur)', 'Lasur Station', 'Late Vasantraoji Dandale Khajgi Krushi Bazar', 'Latur', 'Latur(Murud)', 'Laxmi Sopan Agriculture Produce Marketing Co Ltd', 'Loha', 'Lonand', 'Lonar', 'MS Kalpana Agri Commodities Marketing', 'Mahagaon', 'Maharaja Agresen Private Krushi Utappan Bazar Sama', 'Mahavir Agri Market', 'Mahavira Agricare', 'Mahesh Krushi Utpanna Bazar, Digras', 'Mahur', 'Majalgaon', 'Malegaon', 'Malegaon(Vashim)', 'Malharshree Farmers Producer Co Ltd', 'Malkapur', 'Manchar', 'Mandhal', 'Mangal Wedha', 'Mangaon', 'Mangrulpeer', 'Mankamneshwar Farmar Producer CoLtd Sanchalit Mank', 'Manmad', 'Manora', 'Mantha', 'Manwat', 'Marathawada Shetkari Khajgi Bazar Parisar', 'Maregoan', 'Mauda', 'Mehekar', 'Mohol', 'Morshi', 'Motala', 'Mudkhed', 'Mukhed', 'Mulshi', 'Mumbai', 'Mumbai- Fruit Market', 'Murbad', 'Murtizapur', 'Murud', 'Murum', 'N N Mundhada Agriculture Market Produce', 'Nagpur', 'Naigaon', 'Nampur', 'Nanded', 'Nandgaon', 'Nandgaon Khandeshwar', 'Nandura', 'Nandurbar', 'Narkhed', 'Nashik(Devlali)', 'Nasik', 'Navapur', 'Ner Parasopant', 'Newasa', 'Newasa(Ghodegaon)', 'Nilanga', 'Nira', 'Nira(Saswad)', 'Om Chaitanya Multistate Agro Purpose CoOp Society', 'Pachora', 'Pachora(Bhadgaon)', 'Paithan', 'Palam', 'Palghar', 'Palus', 'Pandhakawada', 'Pandharpur', 'Panvel', 'Parali Vaijyanath', 'Paranda', 'Parbhani', 'Parner', 'Parola', 'Parshiwani', 'Partur', 'Partur(Vatur)', 'Patan', 'Pathardi', 'Pathari', 'Patoda', 'Patur', 'Pavani', 'Pen', 'Perfect Krishi Market Yard Pvt Ltd', 'Phaltan', 'Pimpalgaon', 'Pimpalgaon Baswant(Saykheda)', 'Pombhurni', 'Pratap Nana Mahale Khajgi Bajar Samiti', 'Premium Krushi Utpanna Bazar', 'Pulgaon', 'Pune', 'Pune(Khadiki)', 'Pune(Manjri)', 'Pune(Moshi)', 'Pune(Pimpri)', 'Purna', 'Pusad', 'Rahata', 'Rahuri', 'Rahuri(Songaon)', 'Rahuri(Vambori)', 'Rajura', 'Ralegaon', 'Ramdev Krushi Bazaar', 'Ramtek', 'Rangrao Patil Krushi Utpanna Khajgi Bazar', 'Ratnagiri (Nachane)', 'Raver', 'Raver(Sauda)', 'Risod', 'Sakri', 'Samudrapur', 'Sangamner', 'Sangli', 'Sangli(Phale, Bhajipura Market)', 'Sangola', 'Sangrampur(Varvatbakal)', 'Sant Namdev Krushi Bazar,', 'Satana', 'Satara', 'Savner', 'Selu', 'Sengoan', 'Shahada', 'Shahapur', 'Shantilal Jain Agro', 'Shegaon', 'Shekari Krushi Khajgi Bazar', 'Shetkari Khajgi Bajar', 'Shetkari Khushi Bazar', 'Shevgaon', 'Shevgaon(Bodhegaon)', 'Shirpur', 'Shirur', 'Shivsiddha Govind Producer Company Limited Sanchal', 'Shree Rameshwar Krushi Market', 'Shree Sairaj Krushi Market', 'Shree Salasar Krushi Bazar', 'Shri Gajanan Maharaj Khajagi Krushi Utpanna Bazar', 'Shrigonda', 'Shrigonda(Gogargaon)', 'Shrirampur', 'Shrirampur(Belapur)', 'Sillod', 'Sillod(Bharadi)', 'Sindi', 'Sindi(Selu)', 'Sindkhed Raja', 'Sinner', 'Sironcha', 'Solapur', 'Sonpeth', 'Suragana', 'Tadkalas', 'Taloda', 'Tasgaon', 'Telhara', 'Tiwasa', 'Tuljapur', 'Tumsar', 'Udgir', 'Ulhasnagar', 'Umared', 'Umarga', 'Umari', 'Umarked(Danki)', 'Umarkhed', 'Umrane', 'Vadgaonpeth', 'Vaduj', 'Vadvani', 'Vai', 'Vaijpur', 'Vani', 'Varora', 'Varud', 'Varud(Rajura Bazar)', 'Vasai', 'Vashi New Mumbai', 'Vita', 'Vitthal Krushi Utpanna Bazar', 'Wardha', 'Washi (Dharashiv)', 'Washim', 'Washim(Ansing)', 'Yashika Agro Marketing', 'Yawal', 'Yeola', 'Yeotmal', 'ZariZamini']
commodities = ['Ajwan', 'Arecanut(Betelnut/Supari)', 'Arhar (Tur/Red Gram)(Whole)', 'Arhar Dal(Tur Dal)', 'Bajra(Pearl Millet/Cumbu)', 'Banana', 'Bengal Gram Dal (Chana Dal)', 'Bengal Gram(Gram)(Whole)', 'Bhindi(Ladies Finger)', 'Bitter gourd', 'Black Gram (Urd Beans)(Whole)', 'Black Gram Dal (Urd Dal)', 'Black pepper', 'Bottle gourd', 'Brinjal', 'Cabbage', 'Carrot', 'Cashewnuts', 'Castor Seed', 'Cauliflower', 'Chikoos(Sapota)', 'Chili Red', 'Chilly Capsicum', 'Coconut', 'Coriander(Leaves)', 'Corriander seed', 'Cotton', 'Cowpea (Lobia/Karamani)', 'Cucumbar(Kheera)', 'Cummin Seed(Jeera)', 'Drumstick', 'French Beans (Frasbean)', 'Garlic', 'Ginger(Dry)', 'Ginger(Green)', 'Grapes', 'Green Gram (Moong)(Whole)', 'Green Gram Dal (Moong Dal)', 'Green Peas', 'Guava', 'Jack Fruit', 'Jamun(Narale Hannu)', 'Jowar(Sorghum)', 'Kulthi(Horse Gram)', 'Lentil (Masur)(Whole)', 'Lime', 'Linseed', 'Maize', 'Mango', 'Methi(Leaves)', 'Mustard', 'Neem Seed', 'Niger Seed (Ramtil)', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Pomegranate', 'Potato', 'Pumpkin', 'Raddish', 'Ragi (Finger Millet)', 'Rice', 'Safflower', 'Sesamum(Sesame,Gingelly,Til)', 'Soanf', 'Soyabean', 'Spinach', 'Sugarcane', 'Sunflower', 'Tomato', 'Turmeric', 'Water Melon', 'Wheat']
varieties = ['1009 Kar', '147 Average', '1st Sort', '2nd Sort', 'Average (Whole)', 'Bansi', 'Black', 'Bold', 'DCH-32(Unginned)', 'Deshi Red', 'Deshi White', 'Desi', 'F.A.Q. Bold', 'Full Green', 'Gajjar', 'Green (Whole)', 'H-4(A) 27mm FIne', 'Hapus(Alphaso)', 'Hybrid', 'Jalgaon', 'Jowar ( White)', 'Jowar (Yellow)', 'Kabul Small', 'Kalyan', 'Kesari', 'Khandesh', 'LH-900', 'LRA', 'Local', 'Maharashtra 2189', 'Mogan Medium', 'N-44', 'Niger Seed', 'Other', 'Pole', 'RCH-2', 'Rajapuri', 'Red', 'Sharbati', 'Totapuri', 'Varalaxmi', 'White', 'White Fozi', 'Yellow', 'Yellow (Black)']
agri_seasons = ["Kharif", "Rabi", "Zaid"]
climate_seasons = ["Monsoon", "Post-Monsoon", "Summer", "Winter"]

#list for yeild data
statess = ['Maharashtra']  # Example states
Districtss = ['Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed', 'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli', 'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur', 'Mumbai suburban', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad', 'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli', 'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim', 'Yavatmal', 'latur'] # Example districts
commoditiess =['Ajwain (Carom Seeds)', 'Aloe Vera', 'Arecanut (Betelnut)', 'Arhar/tur', 'Ashwagandha', 'Bajra', 'Bajra (Pearl Millet)', 'Banana', 'Barley', 'Ber (Indian Jujube)', 'Berseem', 'Bitter Gourd', 'Black Pepper', 'Bottle Gourd', 'Brinjal (Eggplant)', 'Cabbage', 'Carrot', 'Cashew Nut', 'Castor Seed', 'Castor seed', 'Cauliflower', 'Chana (Bengal Gram)', 'Chikoo (Sapota)', 'Chilli', 'Cluster Beans (Gavar)', 'Coconut', 'Coffee', 'Coriander', 'Coriander Seeds', 'Cotton', 'Cotton(lint)', 'Cucumber', 'Cumin (Jeera)', 'Custard Apple', 'Dill Seeds', 'Drumstick', 'Fennel (Saunf)', 'Fenugreek (Methi)', 'Fig (Anjeer)', 'French Beans', 'Garlic', 'Ginger', 'Gram', 'Grapes', 'Green Peas', 'Groundnut', 'Guava', 'Hybrid Napier Grass', 'Jackfruit', 'Jamun', 'Jowar', 'Jowar (Sorghum)', 'Kulthi (Horse Gram)', 'Lady Finger (Bhindi)', 'Lemon', 'Lemongrass', 'Linseed', 'Lobia (Cowpea)', 'Lucerne', 'Maize', 'Maize (For Fodder)', 'Mango', 'Masoor (Lentil)', 'Moong (Green Gram)', 'Moong(green gram)', 'Muskmelon', 'Mustard', 'Mustard Seeds', 'Neem', 'Niger (Ramtil)', 'Niger seed', 'Onion', 'Orange', 'Papaya', 'Pineapple', 'Pomegranate', 'Potato', 'Pumpkin', 'Radish', 'Ragi', 'Ragi (Finger Millet)', 'Rajma (Kidney Beans)', 'Rapeseed & Mustard', 'Rice', 'Safflower', 'Safflower (Kardi)', 'Sarpagandha', 'Sesame (Til)', 'Sesamum', 'Sorghum (For Fodder)', 'Soyabean', 'Soybean', 'Spinach', 'Sugarcane', 'Sunflower', 'Sweet Lime (Mosambi)', 'Tea', 'Tobacco', 'Tomato', 'Tulsi (Holy Basil)', 'Tur (Arhar/Red Gram)', 'Turmeric', 'Urad', 'Urad (Black Gram)', 'Watermelon', 'Wheat'] # Example commodities
seasonss =['Kharif', 'Rabi', 'Summer', 'Whole Year']  # Example seasons


# Define a Pydantic model to accept the input data from the user
class YieldEstimationRequest(BaseModel):
    state: str
    district: str
    commodity: str
    season: str
    area_hectare: float  # Area in hectares

# Replace with your Gemini API URL and API Key


 # Replace with your actual API key
class UserMessage(BaseModel):
    query: str

# Initialize the Gemini client
client = genai.Client(api_key="Your Gemini API key Here")

# Function to interact with Gemini API using genai client
def formatResponse(responseText: str) -> str:
    # Replace ***text*** with <h3>text</h3> (for headings)
    formattedResponse = re.sub(r'\*\*\*(.*?)\*\*\*', r'<h3>\1</h3>', responseText)
    
    # Replace **text** with <strong>text</strong> (for bold text)
    formattedResponse = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formattedResponse)
    
    # Replace *text* with <em>text</em> (for italic text)
    formattedResponse = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formattedResponse)
    
    # Replace section headers (e.g., "Crops:") with <h4>Crops:</h4>
    formattedResponse = re.sub(r'(^|\n)\s*([A-Za-z\s]+):', r'<h4>\2:</h4>', formattedResponse)
    
    # Replace bullet points (- item) with <li>item</li>
    formattedResponse = re.sub(r'\n- (.*?)(?=\n|$)', r'<li>\1</li>', formattedResponse)
    
    # Replace newlines with <br> for line breaks
    formattedResponse = re.sub(r'\n', r'<br>', formattedResponse)
    
    # Wrap bullet points in a <ul> tag if any <li> tags are present
    if '<li>' in formattedResponse:
        formattedResponse = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', formattedResponse)
    
    return formattedResponse
# Function to interact with the Gemini API using genai client
def get_gemini_response(user_query: str) -> str:
    try:
        # Prepare the content to ask for agriculture-related responses in bullet points
        prompt = f"Provide short, concise, and straightforward answers related to agriculture in bullet points. Answer the query: {user_query}"

        # Call the Gemini model to generate content based on agriculture-related queries
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Specify your model
            contents=prompt,  # Send the agriculture-related prompt
        )

        # Check if the response has text
        if response.text:
            # Format the response text before returning it
            formatted_response = formatResponse(response.text)
            return formatted_response
        
        return "Sorry, I could not find an answer."
    
    except Exception as e:
        # Handle exceptions (e.g., API errors, connection issues)
        raise HTTPException(status_code=500, detail=f"Error querying Gemini API: {str(e)}")

# Route to handle user queries


# Route for Home
@app.get("/predict")
def home():
    return {"message": "Welcome to Agrosarthi API"}

# Crop Prediction Route
@app.post("/predict/")
def predict(input_data: CropPredictionInput):
    if crop_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([input_data.dict().values()], columns=["N", "P", "K", "temperature", "humidity", "pH", "rainfall"])

        # Make prediction
        predicted_label = crop_model.predict(data)[0]
        reverse_crop_mapping = {
            0: 'aloevera', 1: 'blackpepper', 2: 'chilli', 3: 'garlic', 4: 'ginger', 5: 'groundnut', 6: 'onion',
            7: 'potato', 8: 'soybean', 9: 'sugarcane', 10: 'sunflower', 11: 'tea', 12: 'tobacco', 13: 'tomato',
            14: 'turmeric', 15: 'wheat', 16: 'apple', 17: 'banana', 18: 'blackgram', 19: 'chickpea', 20: 'coconut',
            21: 'coffee', 22: 'cotton', 23: 'grapes', 24: 'jute', 25: 'kidneybeans', 26: 'lentil', 27: 'maize',
            28: 'mango', 29: 'mothbeans', 30: 'mungbean', 31: 'muskmelon', 32: 'orange', 33: 'papaya', 34: 'pigeonpeas',
            35: 'pomegranate', 36: 'rice', 37: 'watermelon'
        }
        predicted_crop = reverse_crop_mapping.get(predicted_label, "Unknown Crop")

        return {"predicted_crop": predicted_crop}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Price Estimation Route
@app.post("/predict-price/")
async def estimate_price(request: PriceEstimationRequest):
    if price_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Prepare the input data for prediction
    input_features = np.array([[request.district, request.month - 1, request.market, request.commodity,
        request.variety, request.agri_season, request.climate_season]])

    try:
        # Predict price using the model
        predicted_price = price_model.predict(input_features)[0]

        result = {
            "district": districts[request.district],
            "month": months[request.month - 1],
            "market": markets[request.market],
            "commodity": commodities[request.commodity],
            "variety": varieties[request.variety],
            "agri_season": agri_seasons[request.agri_season],
            "climate_season": climate_seasons[request.climate_season],
            "predicted_price": predicted_price
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    

@app.post("/predict-yield/")
async def estimate_yield(request: YieldEstimationRequest):
    # Check if model is loaded
    if yield_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    # Validate inputs
    if request.state not in statess:
        raise HTTPException(status_code=400, detail="Invalid state")
    if request.district not in Districtss:
        raise HTTPException(status_code=400, detail="Invalid district")
    if request.commodity not in commoditiess:
        raise HTTPException(status_code=400, detail="Invalid commodity")
    if request.season not in seasonss:
        raise HTTPException(status_code=400, detail="Invalid season")
    
    # Prepare the input features for prediction
    input_features = np.array([[
        statess.index(request.state), 
        Districtss.index(request.district), 
        commoditiess.index(request.commodity), 
        seasonss.index(request.season), 
        request.area_hectare
    ]])

    try:
        # Predict yield using the model (In ton/ha) - Replace this with your actual model prediction logic
        predicted_yield = np.random.random()  # Simulated prediction, replace with actual model prediction

        # Construct the result to return to the frontend
        result = {
            "state": request.state,
            "district": request.district,
            "commodity": request.commodity,
            "season": request.season,
            "area_hectare": request.area_hectare,
            "predicted_yield_ton_ha": predicted_yield  # Returning predicted yield in ton/ha
        }

        return JSONResponse(content=result)

    except Exception as e:
        # Handle any unexpected errors
        return JSONResponse(content={"error": str(e)}, status_code=400)
    
@app.post("/query/")
async def query_chatbot(user_message: UserMessage):
    user_query = user_message.query
    
    # Get response from Gemini API
    response = get_gemini_response(user_query)
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
