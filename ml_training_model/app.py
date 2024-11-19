import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_model(model_path='xgb_model.pkl'):
    """Load the trained XGBoost model"""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def get_categorical_mappings():
    """Define the mappings for categorical variables"""
    mappings = {
        'equipment_type': {'Harvester': 0, 'Tractor': 1, 'Plow': 2, 'Seeder': 3},
        'equipment_name': {'Model A': 0, 'Model B': 1, 'Model C': 2, 'Model D': 3},
        'brand': {'Brand X': 0, 'Brand Y': 1, 'Brand Z': 2},
        'condition': {'Excellent': 0, 'Good': 1, 'Fair': 2},
        'usage_frequency': {'Daily': 0, 'Weekly': 1, 'Monthly': 2},
        'fuel_type': {'Diesel': 0, 'Gasoline': 1, 'Electric': 2},
        'location': {'Urban': 0, 'Suburban': 1, 'Rural': 2},
        'demand_level': {'High': 0, 'Medium': 1, 'Low': 2},
        'rental_duration_preference': {'Short-term': 0, 'Medium-term': 1, 'Long-term': 2},
        'technology_level': {'Basic': 0, 'Advanced': 1, 'Premium': 2},
        'weather_dependency': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    return mappings

def get_user_input(mappings):
    """Get user input for all features"""
    print("\nPlease provide the following information:")
    
    # Categorical inputs
    categorical_inputs = {}
    for feature, mapping in mappings.items():
        while True:
            print(f"\n{feature} options: {', '.join(mapping.keys())}")
            value = input(f"Enter {feature}: ").strip()
            if value in mapping:
                categorical_inputs[feature] = mapping[value]
                break
            print("Invalid input! Please choose from the given options.")

    # Numerical inputs
    numerical_features = {
        'age': 'Enter equipment age (years)',
        'horsepower': 'Enter horsepower',
        'maintenance_score': 'Enter maintenance score (0-10)',
        'fuel_efficiency': 'Enter fuel efficiency score (0-10)',
        'seasonal_demand_multiplier': 'Enter seasonal demand multiplier (0.5-2.0)'
    }
    
    numerical_inputs = {}
    for feature, prompt in numerical_features.items():
        while True:
            try:
                value = float(input(f"{prompt}: "))
                numerical_inputs[feature] = value
                break
            except ValueError:
                print("Please enter a valid number!")

    # Combine all inputs in the correct order
    feature_order = [
        'equipment_type', 'equipment_name', 'brand', 'age', 'condition',
        'usage_frequency', 'fuel_type', 'horsepower', 'maintenance_score',
        'location', 'demand_level', 'rental_duration_preference',
        'fuel_efficiency', 'technology_level', 'weather_dependency',
        'seasonal_demand_multiplier'
    ]
    
    final_inputs = []
    for feature in feature_order:
        if feature in categorical_inputs:
            final_inputs.append(categorical_inputs[feature])
        else:
            final_inputs.append(numerical_inputs[feature])
            
    return np.array([final_inputs])

def main():
    """Main function to run the prediction loop"""
    try:
        # Load the model
        model = load_model()
        mappings = get_categorical_mappings()
        
        while True:
            # Get user input
            input_data = get_user_input(mappings)
            
            # Make prediction
            predicted_price = model.predict(input_data)[0]
            
            print(f"\nPredicted Rental Price: Rupees{predicted_price:.2f} per day")
            
            # Ask if user wants to make another prediction
            if input("\nWould you like to make another prediction? (yes/no): ").lower() != 'yes':
                break
                
    except FileNotFoundError:
        print("Error: Model file 'xgb_model.pkl' not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()