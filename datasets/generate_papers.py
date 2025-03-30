import arxiv
import pandas as pd
import random
from time import sleep
from datetime import datetime

def find_longest_text(df, column_name="Text"):
    """
    Finds the row with the longest text (by word count) in a given DataFrame column.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the text data.
    column_name (str): The column name to search in.
    
    Returns:
    int: The word count of the longest text.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Find the maximum word count
    df["word_count"] = df[column_name].astype(str).apply(lambda x: len(x.split()))
    return df["word_count"].max()

def truncate_text(text, max_words, variation=True):
    """
    Truncates a given text to the maximum word count with optional variation.
    
    Args:
    text (str): The input text.
    max_words (int): The maximum word count allowed.
    variation (bool): If True, some texts will be slightly shorter.

    Returns:
    str: The truncated text.
    """
    words = text.split()
    if variation and random.random() < 0.3:  # 30% chance to shorten text
        max_length = random.randint(int(max_words * 0.7), max_words)  
    else:
        max_length = max_words
    return " ".join(words[:max_length])

# Load dataset to find longest text
file_path = "dataset2_inputs.csv"  # Update with your actual file path
df_existing = pd.read_csv(file_path, sep=";")

# Get the longest text word count
max_word_count = find_longest_text(df_existing)

# List of queries (scientific topics)
queries = [
    "Quantum entanglement", "Black hole thermodynamics", "General relativity and gravitational waves",
    "Dark matter and dark energy", "String theory and extra dimensions", "Exoplanets and habitability",
    "Deep learning in healthcare", "Reinforcement learning in robotics", "Ethical AI and bias in machine learning",
    "Natural language processing advancements", "Large language models and their impact", "AI in scientific discovery",
    "CRISPR and gene editing", "Human microbiome and gut health", "Neuroscience and brain-computer interfaces",
    "Aging and longevity research", "Cancer immunotherapy", "Synthetic biology",
    "Ocean acidification and marine ecosystems", "Renewable energy storage technologies", 
    "Effects of climate change on biodiversity", "Carbon capture and storage", "Microplastics and their environmental impact",
    "Chaos theory and deterministic systems", "Quantum computing and cryptography", 
    "Mathematical modeling in epidemiology", "Topological data analysis in AI",
    "Large Language Models", "Generative AI", "Air polution", "Lung Caner","Lung Diease",
    "Medicine", "Health","Machine Learning","Internet of Things","5G","Quantum Computing","Whales","Dinossaurs","Mummyes",
    "Radiolgy","X-ray","CT Scan","MRI","Ultrassound","Animal Behavior in Wildlife","Animal Behavior in Domesticated Animals",
    "Human mobility and gait analysis","Human behavior in social networks","Human behavior in online communities","Intrusion Detection Systems",
    "Geofencing"
]

# Initialize an empty list to store results
data = []

# Fetch abstracts from arXiv
for query in queries:
    # Create a more specific search with a date range
    search = arxiv.Search(
        query=f"{query}",
        max_results=500,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    count = 0  # Track the number of valid papers
    try:
        for result in search.results():
            # Additional filtering to ensure papers are before 2018
            if result.published.year < 2023 and count < 50:
                
                truncated_text = truncate_text(result.summary, max_word_count)
                data.append({
                    "text": truncated_text, 
                    "source": "human",
                    "year": result.published.year,
                    "title": result.title
                })
                count += 1
            
            if count >= 50:
                break  # Stop after collecting 50 valid papers
        
        sleep(4)  # Avoid API rate limits
        print(f"Query: {query}, Papers collected: {len(data)}")
    except Exception as e:
        print(f"Error fetching papers for query '{query}': {e}")

# Convert to DataFrame and save
df_new = pd.DataFrame(data)
df_new[["text","source"]].to_csv("scientific_texts.csv", index=False)

print("✅ Abstracts fetched, truncated, and saved as 'scientific_texts.csv'.")