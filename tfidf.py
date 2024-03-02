import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Example sentences
sentences = [
    "The car is driven on the road.",
    "The truck is driven on the highway."
]

# Tokenize sentences and remove stopwords
stop_words = set(stopwords.words('english'))
tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
tokenized_sentences = [[word for word in sentence if word not in stop_words] for sentence in tokenized_sentences]

# Join tokenized sentences back into strings
sentences_joined = [' '.join(sentence) for sentence in tokenized_sentences]
print('/n/n')

from sklearn.metrics.pairwise import cosine_similarity
vectorizer = TfidfVectorizer()
tfidf_vectors = vectorizer.fit_transform(sentences_joined)

# Compute pairwise cosine similarity between sentence vectors
cosine_similarities = cosine_similarity(tfidf_vectors)

print("Cosine Similarity Matrix:")
print(cosine_similarities)
print('/n/n')

# # Calculate TF-IDF vectors
# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectors = tfidf_vectorizer.fit_transform(sentences_joined)

# Build nearest neighbors model
nn_model = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
nn_model.fit(tfidf_vectors.toarray())

# Query the model to find the most similar sentence to the first sentence
query_vec = vectorizer.transform([' '.join(tokenized_sentences[0])])
distances, indices = nn_model.kneighbors(query_vec.toarray(), return_distance=True)

# Print the most similar sentence
print("Most similar sentence to '{}': {}".format(sentences[0], sentences[indices[0][1]]))


# Print TF-IDF similarity between first and second sentence
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# tfidf_similarity = cosine_similarity(tfidf_vectors)
# print(tfidf_similarity)
# print("TF-IDF similarity between first and second sentence:", tfidf_similarity[0][1])