import pickle
import numpy as np
import json
from collections import Counter

# Load the trained model and vectorizer
with open("ml_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Load country weights from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)
    country_weights = config.get("country_weights", {})

# Initialize a counter to track how often each country is mentioned
crime_count = Counter()

def predict_priority(crime, country):
    """Predict priority based on crime type and dynamically update country weight."""

    # Transform input text for prediction
    text_vector = vectorizer.transform([crime + " " + country])

    # Get base model prediction
    base_priority = model.predict(text_vector)[0]

    # Convert categorical prediction to numeric priority
    priority_mapping = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
    base_priority = priority_mapping.get(base_priority, 0.5)

    # Get current weight (default 1.0 if country is not listed)
    weight = country_weights.get(country, 1.0)

    # Adjust priority based on weight
    adjusted_priority = np.clip(base_priority * weight, 0, 1)

    # Track how many times each country is mentioned
    crime_count[country] += 1

    return adjusted_priority

def update_country_weights():
    """Automatically update country weights based on frequency of mentions."""
    global country_weights

    # Normalize country occurrences to determine new weights
    max_crime = max(crime_count.values(), default=1)
    for country in crime_count:
        country_weights[country] = 0.5 + (crime_count[country] / max_crime)  # Scale weights

    # Save updated weights back to config.json
    with open("config.json", "w") as config_file:
        json.dump({"country_weights": country_weights}, config_file, indent=4)

    print("✅ Updated country weights in config.json based on latest analysis.")
