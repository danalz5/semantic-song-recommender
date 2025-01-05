import pandas as pd
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import time

# input is texts which are the batch of lyrics to be embedded, the tokenizer and model, and the device (CPU or GPU)
# function is a helper which generates a batch embedding for the given text, in this case lyrics
# the batch embeddings are returned once extracted 
def generateBatchEmbeddings(texts, tokenizer, model, device):
    # max 128 tokens and shorter inputs are padded to 128
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()} # uses GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # extracts the embeddings
    return embeddings

# input is the tokenizer, model, and device (GPU or CPU)
# generated the file that stores the embeddings for all the lyrics
# returns the embeddings
def createEmbeddingsFile(tokenizer, model, device):
    batchSize = 128
    embeddingPath = "data/embeddings.npy"
    df = pd.read_csv("data/cleaned_lyrics.csv")
    lyrics = df['lyrics'].fillna(" ").astype(str).tolist()
    embeddings = []

    for startIndex in range(0, len(lyrics), batchSize):
        batch = lyrics[startIndex:startIndex + batchSize]
        batch_embeddings = generateBatchEmbeddings(batch, tokenizer, model, device)
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings).astype('float32')
    np.save(embeddingPath, embeddings)  # save embeddings to file
    return embeddings

# input are the lyrics embeddings
# creates the faiss index and saves it to a file
# returns nothing
def createFaissIndexFile(embeddings):
    indexPath = "data/faiss_index.bin"
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)  # add embeddings to the index
    faiss.write_index(index, indexPath)  # save index to file
    return

def generateFiles():
    start_time = time.time()
    modelType = "sentence-transformers/all-MiniLM-L6-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelType)
    model = AutoModel.from_pretrained(modelType).to(device)

    embeddings = createEmbeddingsFile(tokenizer, model, device)
    createFaissIndexFile(embeddings)
    end_time = time.time()
    print(end_time - start_time)
    return 0

generateFiles()
