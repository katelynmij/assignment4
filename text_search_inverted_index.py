from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re

documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The medication caused a headache and nausea, but no dizziness was reported.",
    "Headache and dizziness are common effects of this medication.",
    "Nausea was reported as a side effect of the medication."
]

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())  # Fixed regex
    return text

def tokenize(text):
    words = text.split()
    unigrams = words
    bigrams = [' '.join(pair) for pair in zip(words, words[1:])]
    trigrams = [' '.join(trio) for trio in zip(words, words[1:], words[2:])]  # Fixed typo
    return unigrams + bigrams + trigrams

def build_inverted_index(documents):
    preprocessed_docs = [preprocess(doc) for doc in documents]
    tokenized_docs = [tokenize(doc) for doc in preprocessed_docs]
    
    all_tokens = [' '.join(doc) for doc in tokenized_docs]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tokens)
    terms = vectorizer.get_feature_names_out()
    
    inverted_index = defaultdict(list)
    for term_idx, term in enumerate(terms):
        for doc_idx, tfidf_value in enumerate(tfidf_matrix[:, term_idx].toarray()):
            if tfidf_value > 0:  # Include only non-zero values
                inverted_index[term].append({'doc': documents[doc_idx], 'tfidf': tfidf_value[0]})
    
    return inverted_index, vectorizer, tfidf_matrix

inverted_index, vectorizer, tfidf_matrix = build_inverted_index(documents)

def rank_documents(query, inverted_index, vectorizer, tfidf_matrix):
    query = preprocess(query)
    query_vector = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_results = sorted(
        [(documents[i], score) for i, score in enumerate(cosine_similarities) if score > 0],
        key=lambda x: -x[1]
    )
    return ranked_results

queries = {
    "q1": "nausea and dizziness",
    "q2": "effects",
    "q3": "nausea was reported",
    "q4": "dizziness",
    "q5": "the medication"
}

results = {}
for q_id, query in queries.items():
    results[q_id] = rank_documents(query, inverted_index, vectorizer, tfidf_matrix)

results
