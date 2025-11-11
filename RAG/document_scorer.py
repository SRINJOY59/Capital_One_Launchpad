import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple
import time
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

class FastQuerySummaryScorer:
    def __init__(self):
        self.word_pattern = re.compile(r'\b\w+\b')
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'
        }
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.embedding_cache = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        words = self.word_pattern.findall(text.lower())
        return [w for w in words if w not in self.stop_words and len(w) > 2]
    
    def jaccard_similarity(self, query: str, summary: str) -> float:
        query_words = set(self.preprocess_text(query))
        summary_words = set(self.preprocess_text(summary))
        
        if not query_words or not summary_words:
            return 0.0
        
        intersection = len(query_words & summary_words)
        union = len(query_words | summary_words)
        
        return intersection / union if union > 0 else 0.0
    
    def enhanced_cosine_similarity(self, query: str, summary: str) -> float:
        try:
            texts = [query, summary]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            similarity = sk_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def semantic_similarity_score(self, query: str, summary: str) -> float:
        cache_key = f"{query}|{summary}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            query_embedding = self.sentence_model.encode([query])
            summary_embedding = self.sentence_model.encode([summary])
            similarity = sk_cosine(query_embedding, summary_embedding)[0][0]
            self.embedding_cache[cache_key] = similarity
            return similarity
        except:
            return self.enhanced_cosine_similarity(query, summary)
    
    def domain_aware_scoring(self, query: str, summary: str) -> float:
        domain_terms = {
            'agriculture': 3.0, 'farming': 3.0, 'crop': 2.5, 'soil': 2.5,
            'fertilizer': 2.0, 'irrigation': 2.0, 'yield': 2.5, 'harvest': 2.0,
            'planting': 2.0, 'cultivation': 2.0, 'productivity': 2.5,
            'organic': 1.5, 'sustainable': 2.0, 'management': 1.5,
            'precision': 2.0, 'technology': 1.5, 'techniques': 1.5
        }
        
        query_words = self.preprocess_text(query)
        summary_words = self.preprocess_text(summary)
        
        domain_score = 0.0
        for word in set(query_words + summary_words):
            if word in domain_terms:
                domain_score += domain_terms[word]
        
        domain_score = domain_score / max(len(query_words) + len(summary_words), 1)
        base_score = self.enhanced_cosine_similarity(query, summary)
        domain_boost = min(domain_score * 0.3, 0.4)
        
        return min(base_score + domain_boost, 1.0)
    
    def ultimate_hybrid_score(self, query: str, summary: str) -> float:
        semantic_score = self.semantic_similarity_score(query, summary)
        domain_score = self.domain_aware_scoring(query, summary)
        cosine_score = self.enhanced_cosine_similarity(query, summary)
        jaccard_score = self.jaccard_similarity(query, summary)
        
        # Weighted ensemble with decreasing importance:
        # - Semantic (40%): Highest weight as it captures meaning and context best
        # - Domain (30%): Agricultural relevance is crucial for domain-specific queries
        # - Cosine (20%): Traditional TF-IDF similarity provides lexical matching
        # - Jaccard (10%): Simple overlap as baseline, least sophisticated metric
        final_score = (
            0.4 * semantic_score +
            0.3 * domain_score +
            0.2 * cosine_score +
            0.1 * jaccard_score
        )
        
        # Sigmoid transformation to spread scores in [0.2, 0.9] range:
        # - Shifts minimum score to 0.2 (avoiding near-zero scores)
        # - Maps to 0.9 maximum for better discrimination
        # - Steepness factor 6 creates good separation between relevant/irrelevant content
        # - Inflection point 0.4 means scores above 0.4 get boosted significantly
        transformed_score = 0.2 + 0.7 * (1 / (1 + np.exp(-6 * (final_score - 0.4))))
        return transformed_score

    def batch_score_summaries(self, query: str, summaries: List[str], method: str = "ultimate") -> List[Tuple[int, float]]:
        scoring_methods = {
            "jaccard": self.jaccard_similarity,
            "cosine": self.enhanced_cosine_similarity,
            "semantic": self.semantic_similarity_score,
            "domain": self.domain_aware_scoring,
            "ultimate": self.ultimate_hybrid_score
        }
        
        if method not in scoring_methods:
            method = "ultimate"
        
        score_func = scoring_methods[method]
        
        scores = []
        for i, summary in enumerate(summaries):
            score = score_func(query, summary)
            scores.append((i, score))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

def demo_fast_scoring():
    scorer = FastQuerySummaryScorer()
    
    query = "What are the best agricultural practices for crop yield improvement?"
    
    summaries = [
        "This document discusses modern farming techniques including crop rotation, soil management, and fertilizer application for maximizing agricultural productivity.",
        "The text covers livestock management, dairy farming operations, and animal health monitoring systems in agricultural settings.",
        "Research findings on precision agriculture, GPS-guided farming, and automated irrigation systems for optimal crop yields.",
        "Traditional farming methods, organic agriculture, and sustainable practices for environmentally friendly crop production.",
        "Weather patterns, climate change impacts, and seasonal variations affecting agricultural planning and crop selection."
    ]
    
    print(f"\nScoring Results for: '{query}'")
    results = scorer.batch_score_summaries(query, summaries, "ultimate")
    
    for rank, (idx, score) in enumerate(results, 1):
        print(f"{rank}. Score: {score:.3f} - {summaries[idx][:80]}...")

if __name__ == "__main__":
    demo_fast_scoring()

