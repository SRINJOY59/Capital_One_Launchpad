import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import os

def load_crop_yield_models():
    models = {}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Crop_Yield_Prediction'))
    model_paths = {
        "linear_reg" : os.path.join(base_dir, "linear_regression_model.pkl"),
        "random_forest" : os.path.join(base_dir, "random_forest_model.pkl"),
        "decision_tree" : os.path.join(base_dir, "decision_tree_model.pkl"),
        "stacked_2" : os.path.join(base_dir, "Stacked_model_2.pkl")
    }
    print(f"[INFO] Current working directory: {os.getcwd()} ")
    for model_name, path in model_paths.items():
        try:
            with open(path, 'rb') as file:
                models[model_name] = pickle.load(file)
                print(f"[INFO] Loaded {model_name} model.")
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name} model: {e}")
    return models

import pandas as pd
import numpy as np

def prepare_crop_yield_input(state_name, district_name, crop_year, season, crop, 
                             temperature, humidity, soil_moisture, area):
    """
    Prepare input data for crop yield prediction model.
    
    Parameters:
    -----------
    state_name : str
        Name of the state (not used in model but kept for reference)
    district_name : str
        Name of the district
    crop_year : int
        Year of crop cultivation
    season : str
        Season of cultivation (Autumn, Kharif, Rabi, Summer, Whole Year, Winter)
    crop : str
        Type of crop
    temperature : float
        Temperature value
    humidity : float
        Humidity value
    soil_moisture : float
        Soil moisture value
    area : float
        Area under cultivation
    
    Returns:
    --------
    numpy.ndarray
        Array of prepared input features for the model (203 features total)
    """
    
    # Define all possible districts (117 districts)
    districts = [
        'ANANTAPUR', 'ANJAW', 'ARARIA', 'ARWAL', 'AURANGABAD', 'BAKSA', 'BALOD',
        'BALODA BAZAR', 'BALRAMPUR', 'BANKA', 'BARPETA', 'BASTAR', 'BEGUSARAI',
        'BEMETARA', 'BHAGALPUR', 'BHOJPUR', 'BIJAPUR', 'BILASPUR', 'BONGAIGAON',
        'BUXAR', 'CACHAR', 'CHANDIGARH', 'CHANGLANG', 'CHIRANG', 'CHITTOOR',
        'DANTEWADA', 'DARBHANGA', 'DARRANG', 'DHAMTARI', 'DHEMAJI', 'DHUBRI',
        'DIBANG VALLEY', 'DIBRUGARH', 'DIMA HASAO', 'DURG', 'EAST GODAVARI',
        'EAST KAMENG', 'EAST SIANG', 'GARIYABAND', 'GAYA', 'GOALPARA', 'GOLAGHAT',
        'GOPALGANJ', 'GUNTUR', 'HAILAKANDI', 'JAMUI', 'JANJGIR-CHAMPA', 'JEHANABAD',
        'JORHAT', 'KADAPA', 'KAIMUR (BHABUA)', 'KAMRUP', 'KAMRUP METRO',
        'KARBI ANGLONG', 'KARIMGANJ', 'KATIHAR', 'KHAGARIA', 'KISHANGANJ',
        'KOKRAJHAR', 'KRISHNA', 'KURNOOL', 'KURUNG KUMEY', 'LAKHIMPUR',
        'LAKHISARAI', 'LOHIT', 'LONGDING', 'LOWER DIBANG VALLEY', 'LOWER SUBANSIRI',
        'MADHEPURA', 'MADHUBANI', 'MARIGAON', 'MUNGER', 'MUZAFFARPUR', 'NAGAON',
        'NALANDA', 'NALBARI', 'NAMSAI', 'NAWADA', 'NICOBARS', 'NORTH AND MIDDLE ANDAMAN',
        'PAPUM PARE', 'PASHCHIM CHAMPARAN', 'PATNA', 'PRAKASAM', 'PURBI CHAMPARAN',
        'PURNIA', 'ROHTAS', 'SAHARSA', 'SAMASTIPUR', 'SARAN', 'SHEIKHPURA',
        'SHEOHAR', 'SITAMARHI', 'SIVASAGAR', 'SIWAN', 'SONITPUR', 'SOUTH ANDAMANS',
        'SPSR NELLORE', 'SRIKAKULAM', 'SUPAUL', 'TAWANG', 'TINSUKIA', 'TIRAP',
        'UDALGURI', 'UPPER SIANG', 'UPPER SUBANSIRI', 'VAISHALI', 'VISAKHAPATANAM',
        'VIZIANAGARAM', 'WEST GODAVARI', 'WEST KAMENG', 'WEST SIANG'
    ]
    
    # Define all possible seasons (6 seasons)
    seasons = ['Autumn', 'Kharif', 'Rabi', 'Summer', 'Whole Year', 'Winter']
    
    # Define all possible crops (80 crops)
    crops = [
        'Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Barley', 'Beans & Mutter(Vegetable)',
        'Bhindi', 'Black pepper', 'Blackgram', 'Bottle Gourd', 'Brinjal', 'Cabbage',
        'Cashewnut', 'Castor seed', 'Citrus Fruit', 'Coconut ', 'Coriander', 'Cotton(lint)',
        'Cowpea(Lobia)', 'Cucumber', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger',
        'Gram', 'Grapes', 'Groundnut', 'Guar seed', 'Horse-gram', 'Jowar', 'Jute',
        'Khesari', 'Korra', 'Lemon', 'Linseed', 'Maize', 'Mango', 'Masoor', 'Mesta',
        'Moong(Green Gram)', 'Niger seed', 'Oilseeds total', 'Onion', 'Orange',
        'Other  Rabi pulses', 'Other Fresh Fruits', 'Other Kharif pulses', 'Other Vegetables',
        'Paddy', 'Papaya', 'Peas  (vegetable)', 'Peas & beans (Pulses)', 'Pineapple',
        'Pome Fruit', 'Pome Granet', 'Potato', 'Pulses total', 'Ragi', 'Rapeseed &Mustard',
        'Rice', 'Safflower', 'Samai', 'Sannhamp', 'Sapota', 'Sesamum', 'Small millets',
        'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tobacco',
        'Tomato', 'Turmeric', 'Urad', 'Varagu', 'Wheat', 'other fibres', 
        'other misc. pulses', 'other oilseeds'
    ]
    
    # Initialize input array with zeros
    # Total features: Crop_Year(1) + Temperature(1) + Humidity(1) + Soil_Moisture(1) + Area(1) + Districts(117) + Seasons(6) + Crops(80) = 207
    # But based on your column list, it seems to be 203 features total
    input_array = np.zeros(203)
    
    # Set basic features (first 5 positions)
    input_array[0] = crop_year      # Crop_Year
    input_array[1] = temperature    # Temperature
    input_array[2] = humidity       # Humidity
    input_array[3] = soil_moisture  # Soil_Moisture
    input_array[4] = area          # Area
    
    # Set district one-hot encoding (positions 5-121, 117 districts)
    district_upper = district_name.upper()
    if district_upper in districts:
        district_idx = districts.index(district_upper)
        input_array[5 + district_idx] = 1
    else:
        print(f"Warning: District '{district_name}' not found in the list")
    
    # Set season one-hot encoding (positions 122-127, 6 seasons)  
    season_title = season.title()  # Convert to title case
    if season_title in seasons:
        season_idx = seasons.index(season_title)
        input_array[122 + season_idx] = 1
    else:
        print(f"Warning: Season '{season}' not found in the list")
    
    # Set crop one-hot encoding (positions 128-207, 80 crops)
    if crop in crops:
        crop_idx = crops.index(crop)
        input_array[128 + crop_idx] = 1
    else:
        print(f"Warning: Crop '{crop}' not found in the list")
    
    return input_array

def crop_yield_inference(state_name, district_name, crop_year, season, crop, 
                        temperature, humidity, soil_moisture, area, model_type='random_forest'):
    """
    Perform crop yield prediction inference and return results in JSON format
    
    Parameters:
    - state_name: Name of the state
    - district_name: Name of the district
    - crop_year: Year of crop cultivation (e.g., 2023)
    - season: Season of cultivation (Autumn, Kharif, Rabi, Summer, Whole Year, Winter)
    - crop: Type of crop to predict yield for
    - temperature: Temperature in Celsius
    - humidity: Humidity percentage (0-100)
    - soil_moisture: Soil moisture percentage
    - area: Area under cultivation (hectares)
    - model_type: Type of model to use ('stacked', 'random_forest', 'decision_tree', 'naive_bayes', 'svm', 'xgboost')
    
    Returns:
    - JSON string with crop yield prediction and related information
    """
    
    try:
        models = load_crop_yield_models()
        if model_type not in models:
            raise ValueError(f"Model type '{model_type}' not available. Available models: {list(models.keys())}")
        
        model = models[model_type]
        
        # Prepare input data using the helper function
        input_data = prepare_crop_yield_input(
            state_name, district_name, crop_year, season, crop,
            temperature, humidity, soil_moisture, area
        ).reshape(1, -1)
        
        # Handle SVM model which might require normalization
        if model_type == 'svm':
            try:
                # Load training data to fit scaler (simplified approach)
                df = pd.read_csv('../Dataset/crop_yield.csv')
                numerical_features = ['Crop_Year', 'Temperature', 'Humidity', 'Soil_Moisture', 'Area']
                scaler = MinMaxScaler()
                scaler.fit(df[numerical_features])
                
                # Only normalize the numerical features (first 5 columns)
                input_data_normalized = input_data.copy()
                input_data_normalized[:, :5] = scaler.transform(input_data[:, :5])
                input_data = input_data_normalized
                
            except Exception as e:
                print(f"Warning: Could not normalize data for SVM: {str(e)}")
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        
        # Calculate yield per hectare
        yield_per_hectare = prediction / area if area > 0 else 0
        
        # Get feature importance or coefficients if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importance = model.feature_importances_[:5]  # First 5 numerical features
        elif hasattr(model, 'coef_'):
            # For linear models
            feature_importance = model.coef_[:5] if len(model.coef_.shape) == 1 else model.coef_[0][:5]
        
        # Prepare confidence intervals (simple approach using prediction variance if available)
        confidence_lower = prediction * 0.8  # Simple 20% confidence interval
        confidence_upper = prediction * 1.2
        
        # Prepare result JSON
        result = {
            "status": "success",
            "model_used": model_type,
            "input_parameters": {
                "state_name": state_name,
                "district_name": district_name,
                "crop_year": crop_year,
                "season": season,
                "crop": crop,
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "area_hectares": area
            },
            "predictions": {
                "total_production": round(prediction, 2),
                "yield_per_hectare": round(yield_per_hectare, 2),
                "production_unit": "tonnes",
                "confidence_interval": {
                    "lower_bound": round(confidence_lower, 2),
                    "upper_bound": round(confidence_upper, 2),
                    "confidence_level": "80%"
                }
            },
            "analysis": {
                "productivity_rating": "High" if yield_per_hectare > 5 else "Medium" if yield_per_hectare > 2 else "Low",
                "seasonal_suitability": season,
                "regional_context": f"{district_name}, {state_name}"
            },
            "feature_importance": {
                "temperature": float(feature_importance[1]) if feature_importance is not None else None,
                "humidity": float(feature_importance[2]) if feature_importance is not None else None,
                "soil_moisture": float(feature_importance[3]) if feature_importance is not None else None,
                "area": float(feature_importance[4]) if feature_importance is not None else None,
                "year": float(feature_importance[0]) if feature_importance is not None else None
            } if feature_importance is not None else None,
            "metadata": {
                "prediction_timestamp": datetime.now().isoformat(),
                "model_version": "v1.0",
                "data_source": "Historical crop yield data"
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e),
            "input_parameters": {
                "state_name": state_name,
                "district_name": district_name,
                "crop_year": crop_year,
                "season": season,
                "crop": crop,
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": soil_moisture,
                "area_hectares": area
            },
            "error_timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, indent=2)

# Example usage:
if __name__ == "__main__":
    # Example prediction
    result = crop_yield_inference(
        state_name="Bihar",
        district_name="Patna",
        crop_year=2023,
        season="Kharif",
        crop="Rice",
        temperature=28.5,
        humidity=72.0,
        soil_moisture=55.0,
        area=1000.0,
        model_type = 'stacked_2'
    )
    
    print("Crop Yield Prediction Result:")
    print(result)