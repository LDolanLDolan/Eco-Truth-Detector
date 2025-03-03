import pandas as pd

# 📝 Load existing dataset (or create if missing)
try:
    crime_data = pd.read_csv("crime_data.csv")
except FileNotFoundError:
    crime_data = pd.DataFrame(columns=["Crime", "Country", "Priority"])

def update_model(new_crime, new_country):
    """Adds new cases to the dataset and retrains the model dynamically."""

    global crime_data

    # Assign default priority (updated dynamically later)
    new_priority = 0.5  
    country_count = crime_data[crime_data["Country"] == new_country].shape[0]
    if country_count > 5:
        new_priority = 0.8  

    # Append new case
    new_entry = pd.DataFrame({"Crime": [new_crime], "Country": [new_country], "Priority": [new_priority]})
    crime_data = pd.concat([crime_data, new_entry], ignore_index=True)

    # Save updated dataset
    crime_data.to_csv("crime_data.csv", index=False)
    print(f"✅ Updated dataset with: {new_crime} in {new_country}")


