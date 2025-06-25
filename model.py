from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AutocompleteModel:
    def __init__(self, words):
        self.words = words
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        self.word_vectors = self.vectorizer.fit_transform(words)

    def suggest(self, prefix, top_k=5):
        prefix_vec = self.vectorizer.transform([prefix])
        similarities = cosine_similarity(prefix_vec, self.word_vectors).flatten()
        
        # Only include words that start with prefix
        candidates = [
            (i, similarities[i]) 
            for i in range(len(self.words)) 
            if self.words[i].startswith(prefix)
        ]
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [self.words[i] for i, _ in candidates[:top_k]]
