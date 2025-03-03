import unittest
from search_algorithms import bfs_search, regex_search
from text_analysis import analyze_sentiment, word_frequency
from ml_heuristic import predict_priority

class TestSearchAlgorithms(unittest.TestCase):
    def test_bfs_search(self):
        text = "Greenwashing is a growing concern in the USA and UK."
        result = bfs_search(text, "greenwashing")
        self.assertTrue(len(result) > 0)

    def test_regex_search(self):
        text = "Greenwashing is a growing concern in the USA and UK."
        result = regex_search(text, "greenwashing")
        self.assertTrue(len(result) > 0)

class TestTextAnalysis(unittest.TestCase):
    def test_sentiment_analysis(self):
        text = "This company falsely claims to be eco-friendly."
        self.assertLess(analyze_sentiment(text), 0)  # Expecting a negative sentiment

    def test_word_frequency(self):
        text = "greenwashing greenwashing fraud fraud deception"
        freq = word_frequency(text)
        self.assertIn(("greenwashing", 2), freq)
        self.assertIn(("fraud", 2), freq)

class TestMLHeuristic(unittest.TestCase):
    def test_predict_priority(self):
        prediction = predict_priority("Greenwashing", "USA")
        self.assertGreater(prediction, 0.5)  # Expecting a high priority score

if __name__ == '__main__':
    unittest.main()
