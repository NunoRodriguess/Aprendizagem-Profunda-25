import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import traceback
from time import sleep
# Load environment variables
load_dotenv()

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY4")
)

# Free models to cycle through (as of early 2024)
FREE_MODELS = [
    "mistralai/mistral-nemo:free"
]

def generate_ai_text(text, models=FREE_MODELS):
    """
    Rewrite the given text using generative AI models while preserving length.
    
    Args:
        text (str): Original scientific text
        models (list): List of models to try
    
    Returns:
        dict: Contains 'text' and 'source' keys
    """
    # Ensure text is a string and not empty
    if not isinstance(text, str) or not text.strip():
        print(f"Warning: Invalid text input: {text}")
        return {'text': text, 'source': 'original'}
    
    # Estimate desired token count based on original text
    original_length = len(text.split())
    max_tokens = min(max(100, original_length), 500)  # Between 100 and 500 tokens
    
    for model in models:
        sleep(1)
        try:
            print(f"Attempting to generate with model: {model}")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": f"Rewrite the following scientific. Preserve the core scientific meaning. Dont use Markdown Language, just the plain text."
                    },
                    {
                        "role": "user", 
                        "content": text
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # Comprehensive null check
            if response and response.choices and len(response.choices) > 0:
                generated_text = response.choices[0].message.content.strip()
                
                # Additional validation
                if generated_text and len(generated_text.split()) > 10:
                    print(f"Successfully generated text with {model}")
                    print(f"Original length: {original_length} words")
                    print(f"Generated length: {len(generated_text.split())} words")
                    return {
                        'text': generated_text, 
                        'source': 'ai'
                    }
                else:
                    print(f"Generated text too short for {model}")
            
            print(f"No valid response from {model}")
        
        except Exception as e:
            print(f"Error with model {model}:")
            print(traceback.format_exc())
    
    # If all models fail, return original text
    print("All models failed. Returning original text.")
    return {
        'text': text, 
        'source': 'original'
    }

def process_scientific_texts(df):
    """
    Process DataFrame of scientific texts, generating AI versions.
    
    Args:
        df (pd.DataFrame): DataFrame with 'text' column
    
    Returns:
        pd.DataFrame: Updated DataFrame with AI-generated texts
    """
    # Add error handling for DataFrame processing
    try:
        # Use apply with error handling to generate AI texts
        results = df['text'].apply(generate_ai_text)
        
        # Create new DataFrame from results
        processed_df = pd.DataFrame(results.tolist())
        
        return processed_df
    except Exception as e:
        print("Error processing DataFrame:")
        print(traceback.format_exc())
        return df

# Detailed logging
print("Starting script...")

# Load the scientific texts
try:
    scientific_texts = pd.read_csv('scientific_texts.csv')
    reddit_texts = pd.read_csv("askscience_processed.csv")
    scientific_texts = pd.concat([scientific_texts, reddit_texts], ignore_index=True)
    print(f"Loaded {len(scientific_texts)} texts")
except Exception as e:
    print("Error loading CSV:")
    print(traceback.format_exc())
    scientific_texts = pd.DataFrame()

# Process the texts
if not scientific_texts.empty:
    scientific_texts = scientific_texts.iloc[800:].reset_index(drop=True)
    processed_texts = process_scientific_texts(scientific_texts.head(200)) # so 200 de cada vez, depois vejo como faço o resto!

    # Ensure columns are in the right order
    processed_texts = processed_texts[['text', 'source']]

    # Save the processed texts
    processed_texts.to_csv('processed_scientific_texts4.csv', index=False)
    print("Text generation complete. Check processed_scientific_texts.csv")
else:
    print("No texts to process")