import requests
import time
import csv
import re
from html import unescape

def clean_html(text):
    clean_text = re.sub(r'<[^>]+>', '', unescape(text))
    return clean_text.strip()

def count_words(text):
    return len(text.split())

def fetch_stackoverflow_answers():
    base_url = "https://api.stackexchange.com/2.3/answers"
    params = {
        "order": "desc",
        "sort": "creation",
        "site": "stackoverflow",
        "pagesize": 100,
        "page": 1,
        "filter": "withbody"
    }
    
    with open('stackoverflow_answers.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'source'])  # Header
        
        while True:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data.get("items"):
                break
                
            for answer in data["items"]:
                body = clean_html(answer["body"])
                word_count = count_words(body)
                
                if 30 <= word_count <= 200:
                    writer.writerow([body, "human"])  # Direct CSV write
                    
            print(f"Processed page {params['page']} | Quota remaining: {data.get('quota_remaining', 0)}")
            
            if not data.get("has_more") or data.get("quota_remaining", 0) <= 0:
                break
                
            time.sleep(max(data.get("backoff", 1) + 1, 2))  # Minimum 2-second delay
            params["page"] += 1

if __name__ == "__main__":
    fetch_stackoverflow_answers()
    print("CSV file created: stackoverflow_answers.csv")