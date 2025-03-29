import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def scrape_golden_girls_wiki():
    """Scrape Golden Girls character data from Wiki pages"""
    
    # URLs for character pages and St. Olaf story
    urls = {
        'Dorothy Zbornak': 'https://goldengirls.fandom.com/wiki/Dorothy_Zbornak',
        'Blanche Devereaux': 'https://goldengirls.fandom.com/wiki/Blanche_Devereaux',
        'Rose Nylund': 'https://goldengirls.fandom.com/wiki/Rose_Nylund',
        'Sophia Petrillo': 'https://goldengirls.fandom.com/wiki/Sophia_Petrillo',
        'St. Olaf Story': 'https://goldengirls.fandom.com/wiki/Once,_In_St._Olaf'
    }
    
    all_character_data = []
    
    for page_name, url in urls.items():
        print(f"Scraping data for {page_name}...")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract information from the page
        # Here we're getting the main article content
        article_content = soup.find('div', class_='page-content')
        
        if article_content:
            paragraphs = article_content.find_all('p')
            text_content = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean up the text
            text_content = re.sub(r'\[\d+\]', '', text_content)  # Remove citation numbers
            text_content = re.sub(r'\s+', ' ', text_content).strip()  # Normalize whitespace
            
            # Split into chunks of around 500 characters
            chunks = []
            words = text_content.split()
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > 500 and current_length > 0:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                    
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Add the data to our collection
            for i, chunk in enumerate(chunks):
                all_character_data.append({
                    'source': page_name,
                    'chunk_id': f"{page_name.lower().replace(' ', '_')}_{i}",
                    'content': chunk
                })
        
        # Also scrape some quotes
        quotes_url = f"{url}/Quotes"
        quotes_response = requests.get(quotes_url)
        quotes_soup = BeautifulSoup(quotes_response.content, 'html.parser')
        
        quotes_section = quotes_soup.find('div', class_='page-content')
        if quotes_section:
            quotes_lists = quotes_section.find_all('ul')
            for quotes_list in quotes_lists:
                quotes = quotes_list.find_all('li')
                for i, quote in enumerate(quotes):
                    quote_text = quote.get_text().strip()
                    if quote_text:
                        all_character_data.append({
                            'source': page_name,
                            'chunk_id': f"{page_name.lower().replace(' ', '_')}_quote_{i}",
                            'content': f"Quote: {quote_text}"
                        })
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(all_character_data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the data
    df.to_csv('data/golden_girls_data.csv', index=False)
    print(f"Saved {len(df)} chunks of character data")
    
    return df

if __name__ == "__main__":
    scrape_golden_girls_wiki()
