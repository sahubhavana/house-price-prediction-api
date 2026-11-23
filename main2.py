import joblib
import pandas as pd
from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel,Field

from fastapi.middleware.cors import CORSMiddleware
app=FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



model=joblib.load("House-Prediction.joblib")
feature=joblib.load("house_feature.joblib")


class House(BaseModel):
    Area_sqft: float = Field(gt=0)
    Bedrooms: int = Field(ge=1)
    Bathrooms: int = Field(ge=1)
    Floors: int = Field(ge=1)
    Age_of_House: int = Field(ge=0)
    Garage: int = Field(ge=0)
    Location_Rating: float = Field(ge=0)
    Distance_to_City_km: float = Field(ge=0)
#home
@app.get("/")
def home():
    return{
        "message":"India house prediction api",
        "status":"Running",
        "endpoint":"send post request"
    }
    
@app.get('/health')

def health():
    return {
        "status": "Running",
        "model": "RandomForestRegressor",
        "features": feature
        
    }
    
    
    # prediction
    
@app.post("/predict")
def predict(house:House):
    
    try:
        input_data = pd.DataFrame([{
            "Area_sqft": house.Area_sqft,
            "Bedrooms": house.Bedrooms,
            "Bathrooms": house.Bathrooms,
            "Floors": house.Floors,
            "Age_of_House": house.Age_of_House,
            "Garage": house.Garage,
            "Location_Rating": house.Location_Rating,
            "Distance_to_City_km": house.Distance_to_City_km
        }])
                    
            
            
        prediction = model.predict(input_data)[0]
        return {
            "predicted_price": f"₹{prediction:,.0f}",
            "currency": "INR"
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))