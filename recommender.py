import pandas as pd
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# input is the lyrics to generate an embedding for, the tokenizer, model and device
# generates and emebedding of the lyrics of a user inputted song to query the faiss index
# returns the generated embedding
def generateEmbedding(lyrics, tokenizer, model, device):
     # max 128 tokens and shorter inputs are padded to 128
    inputs = tokenizer(lyrics, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()} # uses GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # extracts the embeddings
    return embeddings

# input is the songRow retrieved from the song/lyric database, and the faiss index
# find the 200 "closest" songs from the faiss index
# returns the distance based on the embeddings of the top 200 songs and the index they are located in the database
def retrieveSongs(songRow, index):
    modelType = "sentence-transformers/all-MiniLM-L6-v2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelType)
    model = AutoModel.from_pretrained(modelType).to(device)
    lyrics = songRow['lyrics'].iloc[0]
    embeddingQuery = generateEmbedding(lyrics, tokenizer, model, device)
    distances, indices = index.search(embeddingQuery, 200) # retrieve a lot in case of covers
    return distances, indices

# input is the distances of top 200 songs and indices in which they are located in the dataframe
# compiles the top 3 closest song from the 200 closest songs and filters out any songs that could just be covers
# returns the 3 recommended songs
def compileRecs(distances, indices, df):
    recommendations = []
    for i in range(200): 
        idx = indices[0][i]
        # filter out any song which has a distance "too close"
        # this filters out songs that are covers
        if distances[0][i] > 10.6:
            recommendations.append({
                'title': df.iloc[idx]['title'],
                'artist': df.iloc[idx]['artist']
            })
        if len(recommendations) >= 5:
            break
    return recommendations

def main():
    print("Welcome to the song lyric recommender!")
    print("Loading song database...")
    df = pd.read_csv("data/cleaned_lyrics.csv")
    print("Loading FAISS index...")
    index = faiss.read_index('data/faiss_index.bin')

    while (1):
        songName = input("Enter the name of a song: ").strip()
        artistName = input("Enter the name of the artist: ").strip()

        print("Searching for song")
        songRow = df[(df['title'].str.lower() == songName.lower()) &
                    (df['artist'].str.lower() == artistName.lower())]
        
        if songRow.empty:
            print("The song you entered wasn't found in our database, please enter another.")
            continue
        
        print("Retrieving similar songs...")
        distances, indices = retrieveSongs(songRow, index)
        print("Compiling recommendations...")
        recommendations = compileRecs(distances, indices, df)
        print ("Here's 5 recommendations we found based on semantic analysis of the lyrics!")
        for rec in recommendations:
            print("Title: " + rec["title"] + " by " + rec["artist"])
    return 0

if __name__ == "__main__":
    main()