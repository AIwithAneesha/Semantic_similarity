import pandas as pd
import openai
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity


api_key = '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'
openai.api_key = api_key

sentences = [
    "I like apples and oranges",
    "I enjoy eating fruits",
    "The sky is blue",
    "Apples are delicious"
]


resp = openai.Embedding.create(
    input= sentences,
    engine="text-embedding-ada-002")


embedding_a = resp['data'][0]['embedding']
embedding_b = resp['data'][1]['embedding']
embedding_c = resp['data'][2]['embedding']
embedding_d = resp['data'][3]['embedding']

li = []
for ele in resp['data']:
    li.append(ele["embedding"])


for i in range(len(sentences) - 1):
    for j in range(i + 1, len(resp["data"])):
        print("text similarity percentage between",sentences[i], "and", sentences[j],"is ", np.dot(resp['data'][i]['embedding'],resp['data'][j]['embedding'])*100)

