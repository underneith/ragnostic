import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
import json

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("app/data", exist_ok=True)

def scrape_character_pages():
    """Scrape individual character pages for detailed information."""
    character_data = []
    
    # Main characters and their Fandom wiki pages
    characters = {
        "Dorothy Zbornak": "https://goldengirls.fandom.com/wiki/Dorothy_Zbornak",
        "Blanche Devereaux": "https://goldengirls.fandom.com/wiki/Blanche_Devereaux",
        "Rose Nylund": "https://goldengirls.fandom.com/wiki/Rose_Nylund",
        "Sophia Petrillo": "https://goldengirls.fandom.com/wiki/Sophia_Petrillo"
    }
    
    for char_name, url in characters.items():
        print(f"Scraping data for {char_name}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract the main content
            content_div = soup.find('div', {'class': 'mw-parser-output'})
            
            if content_div:
                paragraphs = content_div.find_all('p')
                
                # Process each paragraph
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 50:  # Skip short paragraphs
                        character_data.append({
                            "character": char_name,
                            "content": text,
                            "source": url
                        })
            
            # Sleep to be kind to Wikipedia
            time.sleep(1)
            
        except Exception as e:
            print(f"Error scraping {char_name}: {e}")
    
    return character_data

def scrape_episode_summaries():
    """Scrape specific episode pages from Golden Girls Fandom Wiki."""
    episode_data = []
    
    # Specific episode URLs from Fandom wiki
    episode_urls = [
        "https://goldengirls.fandom.com/wiki/Yes,_We_Have_No_Havanas",
        "https://goldengirls.fandom.com/wiki/Once,_In_St._Olaf"
    ]
    
    try:
        for episode_url in episode_urls:
            try:
                print(f"Scraping episode data from {episode_url}...")
                response = requests.get(episode_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract episode title
                title_element = soup.find('h1', {'class': 'page-header__title'})
                if title_element:
                    title = title_element.get_text().strip()
                else:
                    title = episode_url.split('/')[-1].replace('_', ' ')
                
                # Look for content in the main article area
                content_div = soup.find('div', {'class': 'mw-parser-output'})
                
                if content_div:
                    # Extract all paragraphs
                    paragraphs = content_div.find_all('p')
                    
                    # Combine paragraph text
                    content = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    
                    # Create episode data entry
                    episode_data.append({
                        "title": title,
                        "content": content,
                        "source": episode_url
                    })
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing episode at {episode_url}: {e}")
        
    except Exception as e:
        print(f"Error scraping episode list: {e}")
    
    return episode_data

def scrape_show_info():
    """Scrape general information about the show."""
    show_data = []
    
    url = "https://goldengirls.fandom.com/wiki/The_Golden_Girls"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get the main content
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        
        if content_div:
            # Process each section
            current_section = "Overview"
            
            for element in content_div.find_all(['h2', 'h3', 'p']):
                if element.name in ['h2', 'h3']:
                    # Update current section
                    section_text = element.get_text().strip()
                    if section_text:
                        current_section = re.sub(r'\[\d+\]', '', section_text).strip()
                elif element.name == 'p':
                    text = element.get_text().strip()
                    if text and len(text) > 50:  # Skip short paragraphs
                        show_data.append({
                            "section": current_section,
                            "content": text,
                            "source": url
                        })
            
            # Also look for specific sections like "About" or "Description"
            about_section = soup.find(lambda tag: tag.name in ['h2', 'h3'] and 'About' in tag.get_text())
            if about_section:
                current = about_section.find_next_sibling()
                while current and current.name != 'h2':
                    if current.name == 'p' and current.get_text().strip():
                        show_data.append({
                            "section": "About",
                            "content": current.get_text().strip(),
                            "source": url
                        })
                    current = current.find_next_sibling()
    
    except Exception as e:
        print(f"Error scraping show info: {e}")
    
    return show_data

def main():
    """Main function to run the scraping process."""
    create_directories()
    
    print("Starting data collection process...")
    
    # Scrape character information
    character_data = scrape_character_pages()
    print(f"Collected {len(character_data)} character data entries")
    
    # Scrape episode summaries
    episode_data = scrape_episode_summaries()
    print(f"Collected {len(episode_data)} episode summaries")
    
    # Scrape general show information
    show_data = scrape_show_info()
    print(f"Collected {len(show_data)} show info entries")
    
    # Combine all data
    all_data = []
    
    for entry in character_data:
        all_data.append({
            "type": "character",
            "title": f"Character: {entry['character']}",
            "content": entry['content'],
            "source": entry['source']
        })
    
    for entry in episode_data:
        all_data.append({
            "type": "episode",
            "title": f"Episode: {entry['title']}",
            "content": entry['content'],
            "source": entry['source']
        })
    
    for entry in show_data:
        all_data.append({
            "type": "show_info",
            "title": f"Show Info: {entry['section']}",
            "content": entry['content'],
            "source": entry['source']
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv("data/raw/golden_girls_data.csv", index=False)
    
    print(f"Data collection complete. Total entries: {len(df)}")
    print(f"Data saved to data/raw/golden_girls_data.csv")

if __name__ == "__main__":
    main()
