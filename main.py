import time
import csv
import os
import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from search_algorithms import bfs_search, regex_search
from text_analysis import analyze_sentiment, word_frequency
from ml_heuristic import predict_priority
from fetch_gdelt_greenwashing_cases import fetch_gdelt_greenwashing_cases
import train_ml_model

# Define a set of countries for detection
COUNTRIES = ["USA", "UK", "Germany", "France", "Canada", "China", "Australia", "Brazil"]

def find_country_mentions(text, search_terms):
    """Finds country mentions linked to wrongdoing by checking if a country appears 
    in the same sentence as a crime-related term."""
    country_counts = Counter()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        for country in COUNTRIES:
            if country in sentence:  # Check if sentence mentions a country
                for term in search_terms:
                    if term in sentence:  # Check if a wrongdoing term is nearby
                        country_counts[country] += 1  # Equal weight to all terms
    
    return country_counts

def generate_csv_report(file_path, search_terms, bfs_results, regex_results, sentiment, word_counts, bfs_time, regex_time, priority_prediction, country_counts):
    """Generates a structured CSV report with clear explanations and verification checks."""

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 🏆 Summary
        writer.writerow(["Report Title", "Text Analysis Results"])
        writer.writerow(["Description", "Summarizes search matches, sentiment analysis, country mentions, and performance."])
        writer.writerow([])

        # ✅ Test Verification
        writer.writerow(["Test Output", "Result"])
        writer.writerow(["Text Length Check", "✅ Passed" if len(word_counts) > 5 else "⚠️ Too Short - May Affect Results"])
        writer.writerow(["Execution Successful", "✅ Yes"])
        writer.writerow([])

        # 🔍 Search Results
        writer.writerow(["Search Terms", ", ".join(search_terms)])
        writer.writerow(["BFS Search Matches", len(bfs_results)])
        writer.writerow(["Regex Search Matches", len(regex_results)])
        writer.writerow([])
        
        # 🕒 Performance Metrics
        writer.writerow(["Performance Metrics", "Time Taken (seconds)"])
        writer.writerow(["BFS Execution Time", f"{bfs_time:.4f}"])
        writer.writerow(["Regex Execution Time", f"{regex_time:.4f}"])
        writer.writerow([])

        # 📉 Sentiment Analysis
        writer.writerow(["Sentiment Analysis", "Score"])
        writer.writerow(["Overall Sentiment Score", f"{sentiment:.3f}"])
        writer.writerow(["Explanation", "Score ranges from -1 (negative) to +1 (positive). Near 0 is neutral."])
        writer.writerow([])

        # 📊 Country Mentions
        writer.writerow(["Country Mentioned", "Times Linked to Crime Terms", "Possible Context"])
        for country, count in country_counts.items():
            context = "Frequent mention - potential greenwashing case" if count > 1 else "Mentioned in passing"
            writer.writerow([country, count, context])
        writer.writerow([])

        # 🏆 Important Words
        writer.writerow(["Top Words", "Occurrence Count"])
        for word, count in word_counts:
            if len(word) > 3:  
                writer.writerow([word, count])
        writer.writerow([])

        # 🔮 Predicted Priority
        writer.writerow(["Predicted Priority", priority_prediction])

def generate_pie_chart(country_counts):
    """Generate a pie chart of country mentions related to greenwashing."""
    if not country_counts:
        print("No country data to plot.")
        return
    
    plt.figure(figsize=(8, 6))
    plt.pie(country_counts.values(), labels=country_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Greenwashing Mentions by Country")
    plt.savefig("greenwashing_pie_chart.png")
    plt.show()

def generate_bar_chart(country_counts):
    """Generate a bar chart for country mentions."""
    if not country_counts:
        print("No country data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    plt.bar(country_counts.keys(), country_counts.values(), color='skyblue')
    plt.xlabel("Country")
    plt.ylabel("Mentions")
    plt.title("Greenwashing Mentions by Country")
    plt.xticks(rotation=45)
    plt.savefig("greenwashing_bar_chart.png")
    plt.show()

def generate_pie_chart_from_csv(csv_path):
    """Generate a pie chart from country_aggregation.csv."""
    try:
        df = pd.read_csv(csv_path)

        # --- Likelihood Conversion ---
        likelihood_mapping = {'High': 3, 'Medium': 2, 'Low': 1}  # Define your mapping
        df['Likelihood_Numerical'] = df['Likelihood'].map(likelihood_mapping)

        # Now use the numerical likelihood for the pie chart
        plt.pie(df['Likelihood_Numerical'], labels=df['Country'], autopct='%1.1f%%', startangle=140)
        plt.title('Country Aggregation by Likelihood')  # Add a title to your chart
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")


def validate_text(text):
    """Ensures the text meets basic constraints before running analysis."""
    if len(text.split()) < 10:
        print("Error: The text is too short for meaningful analysis.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Text Analysis with Multi-Term Search & Country Detection")
    parser.add_argument("--file", type=str, help="Path to the text file for analysis")
    parser.add_argument("--terms", type=str, default="fraud,deception,scandal", help="Comma-separated search terms")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found!")
        return

    output_file = "text_analysis_report.csv"
    search_terms = args.terms.split(",")  # Convert input terms to a list
    
    print("🔍 Fetching real-world greenwashing cases from GDELT...")
    fetch_gdelt_greenwashing_cases()  # ✅ Fetch latest cases before analysis

    print(f"📄 Processing {args.file} with terms {search_terms}...")
    
    with open(args.file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Perform BFS search
    start_time = time.time()
    bfs_results = [bfs_search(text, term) for term in search_terms]
    bfs_time = time.time() - start_time
    
    # Perform Regex search
    start_time = time.time()
    regex_results = [regex_search(text, term) for term in search_terms]
    regex_time = time.time() - start_time
    
    # Perform Sentiment Analysis
    sentiment = analyze_sentiment(text)
    
    # Calculate Word Frequency
    word_counts = word_frequency(text)
    
    
    # Predict search priority
    priority_prediction = predict_priority(search_terms[0], "Global")
    
    # Extract country mentions near search terms
    country_counts = find_country_mentions(text, search_terms)
    
    # Generate CSV Report
    generate_csv_report(output_file, search_terms, bfs_results, regex_results, sentiment, word_counts, bfs_time, regex_time, priority_prediction, country_counts)
    

    # Run the model training script
    train_ml_model.main()

    # Generate Pie and Bar Charts from country_aggregation.csv
    generate_pie_chart_from_csv("country_aggregation.csv")
    

    print(f"✅ Analysis complete. Report saved as {output_file}")

if __name__ == "__main__":
    main()