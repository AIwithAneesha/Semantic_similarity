import spacy

# Load SpaCy model with pre-trained word embeddings
nlp = spacy.load("en_core_web_md")

# Process the sentences to obtain Doc objects
a_doc = nlp("I like apples and oranges")
b_doc = nlp("I enjoy eating fruits")

# Access the vector representations of the entire sentences
a_embedding = a_doc.vector
b_embedding = b_doc.vector

# Calculate the similarity between the embeddings
similarity = a_doc.similarity(b_doc)

# Print the similarity
print("Similarity between the sentences:", similarity)
