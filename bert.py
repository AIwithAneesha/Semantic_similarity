from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Define sentences
sentences = [
    "I like apples and oranges",
    "I enjoy eating fruits",
    "The sky is blue",
    "Apples are delicious"
]

# Query
query = "I like fruits"

# Tokenize sentences and query
tokenized_sentences = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
tokenized_query = tokenizer(query, padding=True, truncation=True, return_tensors='pt')

# Obtain embeddings for sentences and query
with torch.no_grad():
    sentence_outputs = model(**tokenized_sentences)
    query_output = model(**tokenized_query)

# Compute cosine similarity between query and sentences
query_embedding = query_output[0][:, 0, :].numpy()
sentence_embeddings = sentence_outputs[0][:, 0, :].numpy()

cosine_similarities = np.inner(query_embedding, sentence_embeddings)

# Print the cosine similarity scores
print("Cosine Similarity Scores:")
for i, score in enumerate(cosine_similarities[0]):
    print(f"Similarity with '{sentences[i]}': {score}")
