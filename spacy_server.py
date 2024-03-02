
import spacy
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load SpaCy model with pre-trained word embeddings
nlp = spacy.load("en_core_web_md")

# Example sentences
sentences = [
    "I like apples and oranges",
    "I enjoy eating fruits",
    "The sky is blue",
    "Apples are delicious"
]

# Compute sentence embeddings
sentence_embeddings = [nlp(sentence).vector for sentence in sentences]

# Convert sentence embeddings to numpy array
sentence_embeddings = np.array(sentence_embeddings)

# Create NearestNeighbors model
k = 2  # Number of nearest neighbors to find
nn_model = NearestNeighbors(n_neighbors=k, metric='cosine')
nn_model.fit(sentence_embeddings)

# Query for nearest neighbors
query = "I like fruits"
query_embedding = nlp(query).vector.reshape(1, -1)  # Reshape for compatibility with sklearn
distances, indices = nn_model.kneighbors(query_embedding)

# Print nearest neighbors
print("Query:", query)
print("Nearest neighbors:")
for i, index in enumerate(indices[0]):
    print(sentences[index], "- Distance:", distances[0][i])
