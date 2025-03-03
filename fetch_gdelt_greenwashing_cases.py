import requests
import pandas as pd
from update_training_data import update_model  # Auto-update dataset

# Define the GDELT API Endpoint for Greenwashing Cases
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc?query=greenwashing&mode=artlist&format=json&maxrecords=10"

def fetch_gdelt_greenwashing_cases():
    """Fetch recent greenwashing news articles from GDELT API and update dataset."""
    
    try:
        response = requests.get(GDELT_URL, timeout=10)

        if response.status_code == 200:
            data = response.json()

            cases = []
            for article in data.get("articles", []):  # Extract articles
                title = article.get("title", "Unknown Case")
                country = article.get("sourcecountry", "Global")  # Extract country if available
                cases.append((title, country))

            if cases:
                for crime, country in cases:
                    update_model(crime, country)  # Update dataset dynamically
                print(f"✅ {len(cases)} new greenwashing cases added from GDELT!")
            else:
                print("⚠️ No greenwashing cases found in GDELT.")

        else:
            print(f"⚠️ Failed to fetch data. HTTP Status Code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    fetch_gdelt_greenwashing_cases()