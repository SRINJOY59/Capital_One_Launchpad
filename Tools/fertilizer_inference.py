import pickle
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Fertilizer-Recommendation')))

class FertilizerRecommendationInference:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/Fertilizer-Recommendation/Fertilizer_StackedModel.pkl'))
        self.model_path = model_path
        self.model = None
        self.le_soil = None
        self.le_crop = None
        self.le_fertilizer = None
        self.load_model()
    
    def load_model(self):
        """Load the trained stacked model and encoders"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data['model']
                self.le_soil = model_data['le_soil']
                self.le_crop = model_data['le_crop']
                self.le_fertilizer = model_data['le_fertilizer']
            else:
                # Legacy format
                self.model = model_data
                
            print("✅ Model loaded successfully!")
            print(f"Available soil types: {list(self.le_soil.classes_)}")
            print(f"Available crop types: {list(self.le_crop.classes_)}")
            print(f"Available fertilizers: {list(self.le_fertilizer.classes_)}")
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def predict(self, temperature, humidity, moisture, soil_type, crop_type, 
                nitrogen, potassium, phosphorous):
        """
        Make fertilizer recommendation
        
        Args:
            temperature (float): Temperature in Celsius [20-40]
            humidity (float): Humidity percentage [30-75]
            moisture (float): Moisture percentage [25-65]
            soil_type (str): Type of soil (Sandy/Loamy/Black/Red/Clayey)
            crop_type (str): Type of crop
            nitrogen (float): Nitrogen content [0-50]
            potassium (float): Potassium content [0-50]
            phosphorous (float): Phosphorous content [0-50]
        
        Returns:
            dict: Prediction results with fertilizer recommendation and confidence
        """
        try:
            # Encode categorical inputs
            soil_encoded = self.le_soil.transform([soil_type])[0]
            crop_encoded = self.le_crop.transform([crop_type])[0]
            
            # Create feature array in correct order
            features = np.array([[temperature, humidity, moisture, soil_encoded, 
                                crop_encoded, nitrogen, potassium, phosphorous]])
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Get fertilizer name
            fertilizer_name = self.le_fertilizer.inverse_transform([prediction])[0]
            confidence = np.max(probabilities)
            
            # Get top 3 recommendations
            top_3_idx = np.argsort(probabilities)[-3:][::-1]
            top_3_fertilizers = []
            for idx in top_3_idx:
                fert_name = self.le_fertilizer.inverse_transform([idx])[0]
                fert_prob = probabilities[idx]
                top_3_fertilizers.append((fert_name, fert_prob))
            
            return {
                'success': True,
                'recommended_fertilizer': fertilizer_name,
                'confidence': confidence,
                'top_3_recommendations': top_3_fertilizers,
                'input_parameters': {
                    'temperature': temperature,
                    'humidity': humidity,
                    'moisture': moisture,
                    'soil_type': soil_type,
                    'crop_type': crop_type,
                    'nitrogen': nitrogen,
                    'potassium': potassium,
                    'phosphorous': phosphorous
                }
            }
            
        except ValueError as e:
            return {'success': False, 'error': f'Invalid input: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}
    
    def get_available_options(self):
        """Get available soil types, crop types, and fertilizers"""
        return {
            'soil_types': list(self.le_soil.classes_),
            'crop_types': list(self.le_crop.classes_),
            'fertilizer_types': list(self.le_fertilizer.classes_)
        }
    
    def validate_inputs(self, temperature, humidity, moisture, soil_type, crop_type, 
                       nitrogen, potassium, phosphorous):
        """Validate input parameters"""
        errors = []
        
        # Numeric range validations
        if not 20 <= temperature <= 40:
            errors.append("Temperature should be between 20-40°C")
        if not 30 <= humidity <= 75:
            errors.append("Humidity should be between 30-75%")
        if not 25 <= moisture <= 65:
            errors.append("Moisture should be between 25-65%")
        if not 0 <= nitrogen <= 50:
            errors.append("Nitrogen should be between 0-50")
        if not 0 <= potassium <= 50:
            errors.append("Potassium should be between 0-50")
        if not 0 <= phosphorous <= 50:
            errors.append("Phosphorous should be between 0-50")
        
        # Categorical validations
        if soil_type not in self.le_soil.classes_:
            errors.append(f"Invalid soil type. Available: {list(self.le_soil.classes_)}")
        if crop_type not in self.le_crop.classes_:
            errors.append(f"Invalid crop type. Available: {list(self.le_crop.classes_)}")
        
        return errors

def interactive_prediction():
    """Interactive function to get user input and make predictions"""
    try:
        # Initialize the inference model
        predictor = FertilizerRecommendationInference()
        
        print("\n" + "="*60)
        print("🌱 FERTILIZER RECOMMENDATION SYSTEM")
        print("="*60)
        
        # Get available options
        options = predictor.get_available_options()
        
        print("Enter the following details:")
        
        # Get user inputs
        temperature = float(input("Temperature (°C) [20-40]: "))
        humidity = float(input("Humidity % [30-75]: "))
        moisture = float(input("Moisture % [25-65]: "))
        
        print(f"\nAvailable Soil Types: {options['soil_types']}")
        soil_type = input("Soil Type: ").strip()
        
        print(f"\nAvailable Crop Types: {options['crop_types']}")
        crop_type = input("Crop Type: ").strip()
        
        nitrogen = float(input("Nitrogen content [0-50]: "))
        potassium = float(input("Potassium content [0-50]: "))
        phosphorous = float(input("Phosphorous content [0-50]: "))
        
        # Validate inputs
        errors = predictor.validate_inputs(temperature, humidity, moisture, soil_type, 
                                         crop_type, nitrogen, potassium, phosphorous)
        
        if errors:
            print("\n❌ Input validation errors:")
            for error in errors:
                print(f"  - {error}")
            return
        
        # Make prediction
        result = predictor.predict(temperature, humidity, moisture, soil_type, 
                                 crop_type, nitrogen, potassium, phosphorous)
        
        # Display results
        print("\n" + "="*60)
        print("🌱 FERTILIZER RECOMMENDATION RESULTS")
        print("="*60)
        
        if result['success']:
            print(f"✅ Recommended Fertilizer: {result['recommended_fertilizer']}")
            print(f"🎯 Confidence Level: {result['confidence']:.2%}")
            
            print("\n📊 Top 3 Recommendations:")
            for i, (fertilizer, prob) in enumerate(result['top_3_recommendations'], 1):
                print(f"  {i}. {fertilizer}: {prob*100:.2f}%")
                
            print(f"\n📝 Input Summary:")
            inputs = result['input_parameters']
            print(f"  Temperature: {inputs['temperature']}°C")
            print(f"  Humidity: {inputs['humidity']}%")
            print(f"  Moisture: {inputs['moisture']}%")
            print(f"  Soil Type: {inputs['soil_type']}")
            print(f"  Crop Type: {inputs['crop_type']}")
            print(f"  N-P-K: {inputs['nitrogen']}-{inputs['phosphorous']}-{inputs['potassium']}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except ValueError:
        print("❌ Error: Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

def example_predictions():
    predictor = FertilizerRecommendationInference()
    
    examples = [
        {
            'name': 'Cotton in Sandy Soil',
            'params': (28.0, 60.0, 40.0, 'Sandy', 'Cotton', 25, 30, 20)
        },
        {
            'name': 'Wheat in Loamy Soil',
            'params': (22.0, 55.0, 45.0, 'Loamy', 'Wheat', 30, 25, 35)
        },
        {
            'name': 'Rice in Clayey Soil',
            'params': (26.0, 70.0, 55.0, 'Clayey', 'Paddy', 35, 20, 25)
        }
    ]
    
    print("\n🌱 Example Fertilizer Recommendations:")
    print("="*60)
    
    for example in examples:
        print(f"\n📋 Example: {example['name']}")
        result = predictor.predict(*example['params'])
        
        if result['success']:
            print(f"  ✅ Recommended: {result['recommended_fertilizer']}")
            print(f"  🎯 Confidence: {result['confidence']:.2%}")
            print(f"  📊 Top 3: {', '.join([f'{fert}({prob:.1%})' for fert, prob in result['top_3_recommendations']])}")
        else:
            print(f"  ❌ Error: {result['error']}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive prediction")
    print("2. Run example predictions")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            interactive_prediction()
        elif choice == "2":
            example_predictions()
        else:
            print("Invalid choice. Running interactive prediction...")
            interactive_prediction()
            
    except Exception as e:
        print(f"Error: {str(e)}")
