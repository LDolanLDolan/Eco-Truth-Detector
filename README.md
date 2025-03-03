# Eco-Truth-Detector
A machine learning tool that detects potential greenwashing in news stories by analyzing text patterns and assigning likelihood scores by country.
# Eco-Truth-Detector

A machine learning tool that detects potential greenwashing in news stories by analyzing text patterns and assigning likelihood scores by country.

## 📊 Overview

Eco-Truth-Detector analyzes news articles to identify potential greenwashing by countries based on patterns learned from GDELT global news data. The tool uses natural language processing and machine learning to predict the likelihood (High, Medium, Low) that a country mentioned in an article is engaging in greenwashing activities.

## 🔍 How It Works

1. **Data Source**: Trained on GDELT news data capturing greenwashing events worldwide
2. **Learning Process**: Uses TF-IDF to extract features from text and logistic regression to learn associations between:
   - Types of greenwashing language
   - Countries mentioned
   - Priority/severity levels
3. **User Flow**:
   - Users submit news text for analysis
   - Model processes text and identifies countries
   - System outputs greenwashing likelihood assessment

## 🚀 Getting Started

### Prerequisites

```
python 3.8+
pandas
scikit-learn
nltk
matplotlib
streamlit 
```

### Installation

```bash
# Clone the repository
git clone https://github.com/ldolanldolan/Eco-Truth-Detector.git
cd Eco-Truth-Detector

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Example Usage

```python
from Eco-Truth-Detector import Eco-Truth-Detector

detector = Eco-Truth-Detector()
text = """Country X announced its ambitious plan to reach carbon 
neutrality by 2030 while simultaneously approving three new coal plants."""

result = detector.analyze(text)
print(result)
# Output: {'country': 'Country X', 'likelihood': 'High'}
```

## 📊 Model Performance

The current model achieves:
- Precision: 68%
- Recall: 96%
- F1 Score: 98%

## 🔮 Future Improvements

- Improve country detection using Named Entity Recognition
- Add timestamp analysis to track changes in greenwashing patterns over time
- Incorporate more nuanced priority/severity scoring
- Expand training data to include more languages and regions

## 📚 Citation

If you use this tool in your research, please cite:

```
@software{Eco-Truth-Detector,
  author = {L Doolan},
  title = {Eco-Truth-Detector: A Tool for Detecting Greenwashing in News},
  year = {2025},
  url = {https://github.com/ldolanldolan/Eco-Truth-Detector}
}
```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
