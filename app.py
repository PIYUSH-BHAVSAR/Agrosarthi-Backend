from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import requests
from typing import Optional
import tempfile
import google.generativeai as genai
import re
app = FastAPI()

# CORS config
origins = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- #
#     GCS Model Load     #
# ---------------------- #


model_paths = {
    "crop": "https://storage.googleapis.com/agrosarthi-models/Crop_prediction.pkl",
    "price": "https://storage.googleapis.com/agrosarthi-models/crop_price_model.pkl",
    "yield": "https://storage.googleapis.com/agrosarthi-models/yield_prediction_model.pkl"
}

loaded_models = {}

def fetch_model_from_gcs(model_key: str) -> Optional[object]:
    try:
        url = model_paths[model_key]
        response = requests.get(url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            model = joblib.load(tmp_file_path)
            print(f"✅ Loaded '{model_key}' model from GCS.")
            return model
        else:
            print(f"❌ Failed to fetch model {model_key}: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error loading model {model_key} from GCS: {e}")
        return None

loaded_models["crop"] = fetch_model_from_gcs("crop")
loaded_models["price"] = fetch_model_from_gcs("price")
loaded_models["yield"] = fetch_model_from_gcs("yield")

# Districts, Markets, Commodities, etc.
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

class UserMessage(BaseModel):
    query: str

# Initialize the Gemini client
genai.configure(api_key="AIzaSyCajv9_POduBEZwkdppKUPrmaIBZp66SS0")

# ✅ Load the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-1.5-pro" / "gemini-pro-vision" if needed

# ✅ Format response for frontend display
def formatResponse(responseText: str) -> str:
    # Headings
    formattedResponse = re.sub(r'\*\*\*(.*?)\*\*\*', r'<h3>\1</h3>', responseText)
    # Bold
    formattedResponse = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formattedResponse)
    # Italics
    formattedResponse = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formattedResponse)
    # Section headers
    formattedResponse = re.sub(r'(^|\n)\s*([A-Za-z\s]+):', r'<h4>\2:</h4>', formattedResponse)
    # Bullet points
    formattedResponse = re.sub(r'\n- (.*?)(?=\n|$)', r'<li>\1</li>', formattedResponse)
    # Line breaks
    formattedResponse = re.sub(r'\n', r'<br>', formattedResponse)
    # Wrap bullet list
    if '<li>' in formattedResponse:
        formattedResponse = re.sub(r'(<li>.*?</li>)', r'<ul>\1</ul>', formattedResponse)

    return formattedResponse

# ✅ Function to interact with Gemini API
def get_gemini_response(user_query: str) -> str:
    try:
        prompt = f"Provide short, concise, and straightforward answers related to agriculture in bullet points. Answer the query: {user_query}"
        
        # Generate content
        response = model.generate_content(prompt)

        if response.text:
            return formatResponse(response.text)
        
        return "Sorry, I could not find an answer."
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying Gemini API: {str(e)}")

class CropPredictionInput(BaseModel):
    nitrogen: float = Field(..., ge=0, le=140)
    phosphorus: float = Field(..., ge=0, le=140)
    potassium: float = Field(..., ge=0, le=200)
    ph: float = Field(..., ge=0, le=14)
    humidity: float = Field(..., ge=0, le=100)
    rainfall: float = Field(..., ge=0, le=300)
    temperature: float = Field(..., ge=0, le=50)

class PriceEstimationRequest(BaseModel):
    district: int
    month: int
    market: int
    commodity: int
    variety: int
    agri_season: int
    climate_season: int

class YieldPredictionRequest(BaseModel):
    state: int
    district: int
    season: int
    crop: int
    area: float

# ----------------------------- #
#         API Endpoints        #
# ----------------------------- #

@app.get("/")
def index():
    return {"message": "Agrosarthi FastAPI Backend with GCS Model Integration"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-crop")
def predict_crop(data: CropPredictionInput):
    model = loaded_models.get("crop")
    if model is None:
        raise HTTPException(status_code=500, detail="Crop prediction model not loaded.")
    
    input_data = [[
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.ph,
        data.humidity,
        data.rainfall,
        data.temperature
    ]]
    prediction = model.predict(input_data)[0]
    return {"recommended_crop": prediction}

@app.post("/predict-price")
def predict_price(data: PriceEstimationRequest):
    model = loaded_models.get("price")
    if model is None:
        raise HTTPException(status_code=500, detail="Price prediction model not loaded.")
    
    input_data = [[
        data.district,
        data.month,
        data.market,
        data.commodity,
        data.variety,
        data.agri_season,
        data.climate_season
    ]]
    prediction = model.predict(input_data)[0]
    return {"estimated_price": float(prediction)}

@app.post("/predict-yield")
def predict_yield(data: YieldPredictionRequest):
    model = loaded_models.get("yield")
    if model is None:
        raise HTTPException(status_code=500, detail="Yield prediction model not loaded.")
    
    input_data = [[
        data.state,
        data.district,
        data.season,
        data.crop,
        data.area
    ]]
    prediction = model.predict(input_data)[0]
    return {"estimated_yield": float(prediction)}

@app.post("/query/")
async def query_chatbot(user_message: UserMessage):
    user_query = user_message.query
    
    # Get response from Gemini API
    response = get_gemini_response(user_query)
    return {"response": response}