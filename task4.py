import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#download VADER if not already installed
nltk.download("vader_lexicon")

#Load dataset
file_path = r"C:\Users\vaish\OneDrive\Documents\netflix_titles.csv\netflix_titles.csv"
df = pd.read_csv(r"C:\Users\vaish\OneDrive\Documents\netflix_titles.csv\netflix_titles.csv")

#Check if 'description' column exists
if "description" not in df.columns:
    print("Error: 'description' column not found in dataset!")
    exit()

# Apply Sentiment Analysis on 'description'
sia = SentimentIntensityAnalyzer()
df["sentiment"] = df["description"].astype(str).apply(lambda x: "Positive" if sia.polarity_scores(x)["compound"] > 0 else "Negative")

#Print results
print("\n🔹 Sentiment Analysis Results:")
print(df[["title", "description", "sentiment"]].head())

