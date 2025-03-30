import pandas as pd

df = pd.read_csv("askscience.csv", sep=",")

# Rename 'body' column to 'text'
df = df.rename(columns={'body': 'text'})

# Process text column: remove newlines, limit to 200 words
df['text'] = (
    df['text']
    .str.replace('\n', ' ', regex=False)  # Replace newlines with spaces
    .str.split()                          # Split into words
    .str[:200]                            # Keep first 200 words
    .str.join(' ')                        # Rejoin words with spaces
)

# Keep only text column and add source column
df = df[['text']]
df['source'] = 'human'

# Optional: Reset index if needed
df = df.reset_index(drop=True)
df.dropna(inplace=True)
df.to_csv("askscience_processed.csv", index=False)