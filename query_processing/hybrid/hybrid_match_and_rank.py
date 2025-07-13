import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .hybrid_query_process import process_hybrid_query
from storage.vector_storage import load_hyprid_ids
from vectorize.hybrid import generate_hybrid

def match_and_rank_hybrid(query_text: str, dataset_name: str, top_k=10, similarity_threshold=0.3, alpha=0.5):

    emb_results = dict(match_and_rank_embedding(query_text, dataset_name, top_k=top_k))

    tfidf_results = dict(match_and_rank_tfidf(query_text, dataset_name, top_k=top_k))

    all_doc_ids = set(emb_results.keys()).union(tfidf_results.keys())

    combined_scores = {}
    for doc_id in all_doc_ids:
        emb_score = emb_results.get(doc_id, 0.0)
        tfidf_score = tfidf_results.get(doc_id, 0.0)
        combined_score = alpha * emb_score + (1 - alpha) * tfidf_score
        combined_scores[doc_id] = combined_score

    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    filtered = [(doc_id, score) for doc_id, score in ranked if score >= similarity_threshold]

    return OrderedDict(filtered[:top_k])
