# this file is to clean up the original excel sheet by only keeping needed columns and cleaning up
# lyrics to be processed by the recommendation system
# this take a couple minutes to run so only run it when changed are made to the code and a new excel sheet is needed

import pandas as pd
import re
from nltk.corpus import stopwords

# input is a string that contains the path to the csv file
# reads in CSV from specified path into a dataframe
# returns the resulting dataframe
def readCSV (path):
    usableCols = ["title", "tag", "artist", "lyrics", "language"]
    lyricsDF = pd.read_csv(path, usecols=usableCols)
    return lyricsDF

# input is text from dataframe (lyrics)
# removes lyric metadata and newlines
# returns a list to be used as the new lyric column in the df
def cleanLyrics(col):
    new_col = []
    setStopWords = set(stopwords.words("english"))
    for song in col:
        # regex detects new line characters and lyrics meta data contained in []
        result = re.sub(r"\[[^\]]*\]|\n", " ", str(song))
        result = re.sub("   ", " ", result).strip() # fixes extra spaces
        result = re.sub(r"[^\w\s]", "", result) # gets rid of punctuations
        result = result.lower()
        result = ' '.join(word for word in result.split() if word not in setStopWords)
        new_col.append(result)
    return new_col

# no input
# cleans the data from the original csv and creates a new csv to work with
# returns nothing
def cleanCSV ():
    df = readCSV("data/lyrics.csv")
    englishDF = df[df["language"].isin(['en'])] # get only english songs
    cleanedDF = englishDF.dropna().copy() # drop missing values
    cleanedDF["lyrics"] = cleanLyrics(cleanedDF["lyrics"])
    cleanedDF.drop('language', axis=1, inplace=True)
    cleanedDF.to_csv("data/cleaned_lyrics.csv", index=False)
    return

cleanCSV()
