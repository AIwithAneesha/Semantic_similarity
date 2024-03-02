import tensorflow_hub as hub
import tensorflow_text

# Load the Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
embed = hub.load(module_url)

# Define sentences
sentences = [
    "I like apples and oranges",
    "I enjoy eating fruits",
    "The sky is blue",
    "Apples are delicious"
]

# Query
query = "I like fruits"

# Compute embeddings for the sentences and the query
sentence_embeddings = embed(sentences)
query_embedding = embed([query])

# Calculate cosine similarity between the query embedding and sentence embeddings
cosine_similarities = np.inner(query_embedding, sentence_embeddings)

# Print the cosine similarity scores
print("Cosine Similarity Scores:")
for i, score in enumerate(cosine_similarities[0]):
    print(f"Similarity with '{sentences[i]}': {score}")
