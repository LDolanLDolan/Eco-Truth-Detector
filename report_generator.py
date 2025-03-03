import csv

def generate_csv_report(file_path, search_term, bfs_results, regex_results, sentiment, word_counts, bfs_time, regex_time, priority_prediction):
    """Generate a CSV report with analysis results."""
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Search Term", search_term])
        writer.writerow(["BFS Search Results", len(bfs_results)])
        writer.writerow(["Regex Search Results", len(regex_results)])
        writer.writerow(["BFS Execution Time", f"{bfs_time:.4f} sec"])
        writer.writerow(["Regex Execution Time", f"{regex_time:.4f} sec"])
        writer.writerow(["Sentiment Score", f"{sentiment:.3f}"])
        writer.writerow(["Top Words", "Word - Count"])
        for word, count in word_counts:
            writer.writerow([word, count])
        writer.writerow(["Predicted Priority", priority_prediction])
