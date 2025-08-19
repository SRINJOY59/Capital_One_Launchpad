import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Crop-Recommendation')))

def load_crop_recommendation_models():
    models = {}
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Crop-Recommendation'))
    model_paths = {
        'decision_tree': os.path.join(base_dir, 'DecisionTree.pkl'),
        'naive_bayes': os.path.join(base_dir, 'NBClassifier.pkl'), 
        'random_forest': os.path.join(base_dir, 'RandomForest.pkl'),
        'svm': os.path.join(base_dir, 'SVMClassifier.pkl'),
        'xgboost': os.path.join(base_dir, 'XGBoost.pkl'),
        'stacked': os.path.join(base_dir, 'StackedModel.pkl')
    }
    print(f"[INFO] Current working directory: {os.getcwd()}")
    for model_name, path in model_paths.items():
        try:
            with open(path, 'rb') as file: models[model_name] = pickle.load(file)
            print(f"[INFO] Loaded {model_name} model")
        except Exception as e: print(f"[ERROR] Failed to load {model_name}: {str(e)}")

    return models

def crop_recommendation_inference(N, P, K, temperature, humidity, ph, rainfall, model_type='random_forest'):
    """
    Perform crop recommendation inference and return top 5 recommendations in JSON format
    
    Parameters:
    - N: Nitrogen content in soil (0-140)
    - P: Phosphorus content in soil (0-145) 
    - K: Potassium content in soil (0-205)
    - temperature: Temperature in Celsius (0-45)
    - humidity: Humidity percentage (0-100)
    - ph: pH value of soil (0-14)
    - rainfall: Rainfall in mm (0-300)
    - model_type: Type of model to use ('stacked', 'random_forest', 'decision_tree', 'naive_bayes', 'svm', 'xgboost')
    
    Returns:
    - JSON string with top 5 crop recommendations and their confidence scores
    """
    
    try:
        models = load_crop_recommendation_models()
        if model_type not in models: raise ValueError(f"Model type '{model_type}' not available. Available models: {list(models.keys())}")
        
        model = models[model_type]
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Handle SVM model which requires normalization
        if model_type == 'svm':
            # For SVM, we need to normalize the data
            # Load training data to fit scaler (this is a simplified approach)
            try:
                df = pd.read_csv('../Dataset/crop_recommendation.csv')
                features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
                scaler = MinMaxScaler()
                scaler.fit(features)
                input_data = scaler.transform(input_data)
            except Exception as e:
                raise ValueError(f"Failed to normalize data for SVM: {str(e)}")
        
        # Get prediction and probabilities
        prediction = model.predict(input_data)[0]
        
        # Get probability scores for all classes
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_data)[0]
            
            # Get top 5 predictions with their probabilities
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                # For XGBoost, we need to get the unique labels from dataset
                try:
                    df = pd.read_csv('../Dataset/crop_recommendation.csv')
                    classes = sorted(df['label'].unique())
                except Exception as e:
                    # Fallback crop list if dataset is not available
                    classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
            
            top_5_crops = [classes[idx] for idx in top_5_idx]
            top_5_probs = [float(probabilities[idx]) for idx in top_5_idx]
            
        else:
            # For models without predict_proba, use decision function or just return the prediction
            top_5_crops = [prediction]
            top_5_probs = [1.0]
            
            # Fill remaining slots with other common crops from dataset
            try:
                df = pd.read_csv('../Dataset/crop_recommendation.csv')
                all_crops = df['label'].unique()
            except Exception as e:
                # Fallback crop list if dataset is not available
                all_crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
            
            other_crops = [crop for crop in all_crops if crop != prediction][:4]
            top_5_crops.extend(other_crops)
            top_5_probs.extend([0.0] * len(other_crops))
        
        # Prepare result JSON
        result = {
            "status": "success",
            "model_used": model_type,
            "input_parameters": {
                "nitrogen": N,
                "phosphorus": P, 
                "potassium": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            },
            "predictions": {
                "recommended_crop": prediction,
                "top_5_recommendations": [
                    {
                        "crop": crop,
                        "confidence_score": prob,
                        "confidence_percentage": round(prob * 100, 2)
                    }
                    for crop, prob in zip(top_5_crops, top_5_probs)
                ]
            },
            "metadata": {
                "total_classes": len(classes) if 'classes' in locals() else "unknown",
                "prediction_timestamp": datetime.now().isoformat()
            }
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e),
            "input_parameters": {
                "nitrogen": N,
                "phosphorus": P,
                "potassium": K, 
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            }
        }
        return json.dumps(error_result, indent=2)

def get_all_model_predictions(N, P, K, temperature, humidity, ph, rainfall):
    """
    Get predictions from all available models and return comparison in JSON format
    """
    models = load_crop_recommendation_models()
    all_predictions = {}
    
    for model_name in models.keys():
        try:
            prediction_json = crop_recommendation_inference(
                N, P, K, temperature, humidity, ph, rainfall, model_type=model_name
            )
            prediction_data = json.loads(prediction_json)
            if prediction_data["status"] == "success":
                all_predictions[model_name] = {
                    "recommended_crop": prediction_data["predictions"]["recommended_crop"],
                    "top_5": prediction_data["predictions"]["top_5_recommendations"]
                }
        except Exception as e:
            all_predictions[model_name] = {"error": str(e)}
    
    result = {
        "status": "success",
        "input_parameters": {
            "nitrogen": N,
            "phosphorus": P,
            "potassium": K,
            "temperature": temperature, 
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall
        },
        "model_predictions": all_predictions,
        "consensus_prediction": get_consensus_prediction(all_predictions)
    }
    
    return json.dumps(result, indent=2)

def get_consensus_prediction(predictions):
    """
    Get consensus prediction from multiple models
    """
    crop_votes = {}
    for model_name, prediction in predictions.items():
        if "recommended_crop" in prediction:
            crop = prediction["recommended_crop"]
            crop_votes[crop] = crop_votes.get(crop, 0) + 1
    
    if crop_votes:
        consensus_crop = max(crop_votes, key=crop_votes.get)
        return {
            "consensus_crop": consensus_crop,
            "vote_count": crop_votes[consensus_crop],
            "total_models": len([p for p in predictions.values() if "recommended_crop" in p]),
            "all_votes": crop_votes
        }
    else:
        return {"consensus_crop": "No consensus", "vote_count": 0}

def validate_input_parameters(N, P, K, temperature, humidity, ph, rainfall):
    """
    Validate input parameters for crop recommendation
    """
    validations = {
        'N': (N, 0, 140, "Nitrogen content should be between 0-140"),
        'P': (P, 0, 145, "Phosphorus content should be between 0-145"), 
        'K': (K, 0, 205, "Potassium content should be between 0-205"),
        'temperature': (temperature, 0, 45, "Temperature should be between 0-45°C"),
        'humidity': (humidity, 0, 100, "Humidity should be between 0-100%"),
        'ph': (ph, 0, 14, "pH should be between 0-14"),
        'rainfall': (rainfall, 0, 300, "Rainfall should be between 0-300mm")
    }
    
    errors = []
    for param_name, (value, min_val, max_val, message) in validations.items():
        try:
            value = float(value)
            if not (min_val <= value <= max_val):
                errors.append(f"{param_name}: {message}, got {value}")
        except (ValueError, TypeError):
            errors.append(f"{param_name}: Must be a valid number, got {value}")
    
    return errors

def get_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall, model_type='stacked', return_format='json'):
    """
    Main function to get crop recommendations with input validation and error handling
    
    Parameters:
    - N: Nitrogen content in soil (0-140)
    - P: Phosphorus content in soil (0-145) 
    - K: Potassium content in soil (0-205)
    - temperature: Temperature in Celsius (0-45)
    - humidity: Humidity percentage (0-100)
    - ph: pH value of soil (0-14)
    - rainfall: Rainfall in mm (0-300)
    - model_type: Type of model to use ('stacked', 'random_forest', 'decision_tree', 'naive_bayes', 'svm', 'xgboost', 'all')
    - return_format: 'json' for JSON string, 'dict' for Python dictionary
    
    Returns:
    - JSON string or dictionary with top 5 crop recommendations and their confidence scores
    """
    
    try:
        # Convert inputs to float
        N = float(N)
        P = float(P)
        K = float(K)
        temperature = float(temperature)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)
        
        # Validate input parameters
        validation_errors = validate_input_parameters(N, P, K, temperature, humidity, ph, rainfall)
        if validation_errors:
            error_result = {
                "status": "error",
                "error_type": "validation_error",
                "error_message": "Input validation failed",
                "validation_errors": validation_errors,
                "input_parameters": {
                    "nitrogen": N,
                    "phosphorus": P,
                    "potassium": K,
                    "temperature": temperature,
                    "humidity": humidity,
                    "ph": ph,
                    "rainfall": rainfall
                }
            }
            return json.dumps(error_result, indent=2) if return_format == 'json' else error_result
        
        # If model_type is 'all', get predictions from all models
        if model_type == 'all':
            result_json = get_all_model_predictions(N, P, K, temperature, humidity, ph, rainfall)
            return result_json if return_format == 'json' else json.loads(result_json)
        
        # Get prediction from specific model
        result_json = crop_recommendation_inference(N, P, K, temperature, humidity, ph, rainfall, model_type)
        result = json.loads(result_json)
        
        # Add additional metadata
        if result["status"] == "success":
            result["metadata"]["function_version"] = "1.0"
            result["metadata"]["available_models"] = ['stacked', 'random_forest', 'decision_tree', 'naive_bayes', 'svm', 'xgboost']
            
        return json.dumps(result, indent=2) if return_format == 'json' else result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "error_type": "runtime_error",
            "error_message": str(e),
            "input_parameters": {
                "nitrogen": N if 'N' in locals() else None,
                "phosphorus": P if 'P' in locals() else None,
                "potassium": K if 'K' in locals() else None,
                "temperature": temperature if 'temperature' in locals() else None,
                "humidity": humidity if 'humidity' in locals() else None,
                "ph": ph if 'ph' in locals() else None,
                "rainfall": rainfall if 'rainfall' in locals() else None
            },
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(error_result, indent=2) if return_format == 'json' else error_result

def quick_predict(N, P, K, temperature, humidity, ph, rainfall):
    """
    Quick prediction function that returns just the top recommendation
    """
    try:
        result = get_crop_recommendation(N, P, K, temperature, humidity, ph, rainfall, return_format='dict')
        
        if result["status"] == "success":
            return {
                "recommended_crop": result["predictions"]["recommended_crop"],
                "confidence": result["predictions"]["top_5_recommendations"][0]["confidence_percentage"],
                "model_used": result["model_used"]
            }
        else:
            return {"error": result.get("error_message", "Unknown error")}
    except Exception as e:
        return {"error": str(e)}

# Example usage and testing function
def test_crop_recommendation():
    """
    Test function to demonstrate usage
    """
    print("=== Testing Crop Recommendation System ===\n")
    
    # Test case 1: Valid inputs
    print("Test 1: Valid inputs")
    result = get_crop_recommendation(90, 42, 43, 20.88, 82.0, 6.5, 202.9)
    print(result)
    print("\n" + "="*80 + "\n")
    
    # Test case 2: Invalid inputs
    print("Test 2: Invalid inputs (temperature too high)")
    result = get_crop_recommendation(90, 42, 43, 50, 82.0, 6.5, 202.9)
    print(result)
    print("\n" + "="*80 + "\n")
    
    # Test case 3: Quick prediction
    print("Test 3: Quick prediction")
    quick_result = quick_predict(83, 45, 60, 28, 70.3, 7.0, 150.9)
    print(json.dumps(quick_result, indent=2))
    print("\n" + "="*80 + "\n")
    
    # Test case 4: All models comparison
    print("Test 4: All models comparison")
    all_models_result = get_crop_recommendation(104, 18, 30, 23.6, 60.3, 6.7, 140.9, model_type='all')
    print(all_models_result)

if __name__ == "__main__":
    test_crop_recommendation()