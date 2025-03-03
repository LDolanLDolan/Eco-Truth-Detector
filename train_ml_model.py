import re
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

def clean_text(value: str) -> str:
    """Clean unwanted symbols from a text field."""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace('\0', '')
    value = re.sub(r'[,\#@]', '', value)
    value = ''.join(ch for ch in value if ch.isprintable())
    return value

def main():
    """main function that runs the model training."""
    CSV_FILE_PATH = "crime_data.csv"
    REPORT_CSV_PATH = "text_analysis_report.csv"
    MODEL_FILE = "greenwashing_model.pkl"
    VECTORIZER_FILE = "greenwashing_vectorizer.pkl"
    COUNTRY_AGGREGATION_CSV = "country_aggregation.csv"

    try:
        print("Current working directory:", os.getcwd())

        try:
            with open(MODEL_FILE, 'rb') as model_file:
                model = pickle.load(model_file)
            with open(VECTORIZER_FILE, 'rb') as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)
                print("Loaded existing model and vectorizer.")
        except FileNotFoundError:
            model = LogisticRegression(max_iter=1000)
            vectorizer = TfidfVectorizer(max_features=1000)
            print("No existing model or vectorizer found. Training from scratch.")

        with open(CSV_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as csv_file, \
             open("corrupted_rows.log", 'w') as log_file:
            header_line = csv_file.readline()
            if not header_line:
                print("Error: CSV file is empty or could not be read.")
                sys.exit(1)
            header_line = header_line.lstrip('\ufeff').replace('\0', '').strip()
            header_fields = [clean_text(h) for h in header_line.split(',')]
            expected_cols = len(header_fields)
            data_rows = []
            line_num = 1
            for raw_line in csv_file:
                line_num += 1
                cleaned_line = raw_line.replace('\0', '').strip()
                if cleaned_line == "":
                    log_file.write(f"Line {line_num}: Empty or whitespace line skipped.\n")
                    continue
                fields = cleaned_line.split(',')
                if len(fields) != expected_cols:
                    if len(fields) > expected_cols:
                        merged_field = ','.join(fields[expected_cols-1:])
                        fields = fields[:expected_cols-1] + [merged_field]
                    elif len(fields) < expected_cols:
                        fields += [''] * (expected_cols - len(fields))
                if len(fields) != expected_cols:
                    raw_line_content = raw_line.strip('\n')
                    log_file.write(
                        f"Line {line_num}: Skipped due to column mismatch "
                        f"(expected {expected_cols}, got {len(fields)}). "
                        f"Raw data: {raw_line_content}\n"
                    )
                    continue
                fields = [clean_text(value) for value in fields]
                data_rows.append(fields)

            if not data_rows:
                print("Warning: No valid data rows found in CSV file.")
            else:
                print(f"Successfully read {len(data_rows)} data rows from CSV.")
                df = pd.DataFrame(data_rows, columns=header_fields)

                print("DataFrame Columns:", df.columns)

                df['Priority'] = pd.to_numeric(df['Priority'], errors='coerce')
                df = df.dropna(subset=['Priority'])
                df = df.reset_index(drop=True)
                print("DataFrame shape after dropna:", df.shape)

                def priority_label(val):
                    if val < 0.4:
                        return 'Low'
                    elif val < 0.7:
                        return 'Medium'
                    else:
                        return 'High'

                y = df['Priority'].apply(priority_label)
                y.name = 'PriorityLabel'

                vectorizer.fit(df['Crime'])
                crime_features = vectorizer.transform(df['Crime']).toarray()
                crime_feature_df = pd.DataFrame(crime_features, columns=vectorizer.get_feature_names_out())

                country_features = pd.get_dummies(df['Country'], prefix='Country')

                X = pd.concat([crime_feature_df, country_features], axis=1)

                print(f"X dataframe shape: {X.shape}")

                if X.empty or y.empty:
                    print("Error: X or y dataframe is empty.")
                    sys.exit(1)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)

                with open(MODEL_FILE, 'wb') as model_file:
                    pickle.dump(model, model_file)
                with open(VECTORIZER_FILE, 'wb') as vectorizer_file:
                    pickle.dump(vectorizer, vectorizer_file)
                print("Saved trained model and vectorizer.")

                y_pred = model.predict(X)

                countries = df['Country'].unique()
                predictions = dict(zip(countries, y_pred))

                try:
                    country_agg_df = pd.read_csv(COUNTRY_AGGREGATION_CSV)
                    print("country_agg_df loaded")
                except FileNotFoundError:
                    country_agg_df = pd.DataFrame(columns=['Country', 'Likelihood'])
                    print("country_agg_df created")

                print(f"country_agg_df: {country_agg_df.head()}")

                for country, prediction in predictions.items():
                    print(f"adding country: {country}")
                    if country in country_agg_df['Country'].values:
                        country_agg_df.loc[country_agg_df['Country'] == country, 'Likelihood'] = prediction
                    else:
                        new_row = pd.DataFrame({'Country': [country], 'Likelihood': [prediction]})
                        country_agg_df = pd.concat([country_agg_df, new_row], ignore_index=True)

                print(f"Saving country aggregation to: {COUNTRY_AGGREGATION_CSV}")
                country_agg_df.to_csv(COUNTRY_AGGREGATION_CSV, index=False)
                print(f"Country likelihoods aggregated in {COUNTRY_AGGREGATION_CSV}")

                y_test_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_test_pred)
                print(f"\nAccuracy: {accuracy:.2f}")
                print("Classification Report:")
                print(classification_report(y_test, y_test_pred))

    except FileNotFoundError:
        print(f"Error: A required file was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

    test_df = pd.DataFrame({"test":[1,2,3]})
    test_df.to_csv("test.csv")

if __name__ == "__main__":
    main()