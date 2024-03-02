from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Example sentences
sentences = [
    "I like apples and oranges",
    "I enjoy eating fruits",
    "The sky is blue",
    "Apples are delicious"
]

# Query
query = "I like fruits"

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the sentences to obtain count vectors
sentence_vectors = vectorizer.fit_transform(sentences)

# Transform the query to obtain its count vector
query_vector = vectorizer.transform([query])

# Create NearestNeighbors model
k = 2  # Number of nearest neighbors to find
nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
nn_model.fit(sentence_vectors)

# Query for nearest neighbors
distances, indices = nn_model.kneighbors(query_vector)

# Print nearest neighbors
print("Query:", query)
print("Nearest neighbors:")
for i, index in enumerate(indices[0]):
    print(sentences[index], "- Distance:", distances[0][i])

