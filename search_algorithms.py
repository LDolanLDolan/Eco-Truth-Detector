import re
from collections import deque

def bfs_search(text, search_term):
    """Perform a BFS-like search for a given word in the text."""
    words = text.split()
    queue = deque([(i, words[i]) for i in range(len(words))])
    results = []
    
    while queue:
        index, word = queue.popleft()
        if search_term.lower() in word.lower():
            results.append((index, word))
    
    return results

def regex_search(text, search_term):
    """Perform regex-based search for efficiency comparison."""
    pattern = re.compile(rf'\b{re.escape(search_term)}\b', re.IGNORECASE)
    return [(m.start(), m.group()) for m in pattern.finditer(text)]
