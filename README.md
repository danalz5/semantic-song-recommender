# semantic-song-recommender

I created this song recommender that recommends songs based on a semantic analysis of the lyrics of a queried song. Unfortunately github doesn't allow me to upload the data files since they're a couple of gigs but here's some steps to set it up.

## Getting Started Steps
### Prerequisite Step:
set up a conda environment to install modules required to run the program
### 1. Getting the database:
I built this recommender using a dataset from kaggle that contains the data of music scraped from the Genius website. Heres the link to that: https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information/discussion?sort=hotness
Download this database and place it into the data folder.

### 2. Cleaning the database:
All you have to do is run the script located at util/db_cleaner.py. This will create a csv filled with the cleaned song lyrics and only required columns. NOTE: this might take you a couple minutes but once you run it you don't have to run it again unless you want to edit the script.

### 3. Creating the embedding and FAISS index files:
All you have to do is run the script located at util/embedding_gen.py. After setting up CUDA and having the script use my GPU this took me ~45 mins (I brought it down to 45 mins after calculating that it originally would take 2 days), could be shorter or longer on your device. Again this only has to be done once unless you want to change the scripts for any reason. NOTE: If you change the CSVs in any way you might have to rerun this to create the files again or else the FAISS index WILL NOT line up with the df indices.

### 4. Run the recommender:
Once all the files are created you can run the recommender script which is recommender.py! Note: Sometimes this will give you songs from completely unknown artists due to the low quality of the original database, I suggest changing the amount of recommendations to more than five on line 46 of this script if you want to get recommendations of songs from more known/mainstream artists.

### Endnote:
I created this project just to teach myself some new stuff and make something cool, hope you enjoy it!
