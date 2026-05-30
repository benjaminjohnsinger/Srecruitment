import requests
from bs4 import BeautifulSoup
import pdfplumber
import io
import re
from datetime import datetime
import csv
import os

def get_pdf_links(html_file_path):
    """Parses the local HTML file and extracts links ONLY from VOAT tables."""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    pdf_links = []
    
    # Locate only the tables designated for 'Volume of Air Traffic'
    voat_tables = soup.find_all('table', summary=lambda s: s and "LAX Volume of Air Traffic" in s)
    
    # Loop through those specific tables and extract every link
    for table in voat_tables:
        for a_tag in table.find_all('a', href=True):
            href = a_tag['href']
            
            # Create full URL for relative paths like /media/13448
            full_url = "https://www.lawa.org" + href if href.startswith('/') else href
            
            if full_url not in pdf_links:
                pdf_links.append(full_url)
                
    return pdf_links

def extract_arrivals_from_pdf(pdf_url):
    """Downloads the VOAT PDF and extracts the reporting Date and Scheduled Arrivals."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(pdf_url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Status {response.status_code}")
    
    # pdfplumber reads the bytes directly, so it doesn't matter if the URL ends in .pdf
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()
        
        # 1. Dynamically find the report month/year
        month_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', text, re.IGNORECASE)
        if not month_match:
            raise Exception("Could not find the reporting month and year in the PDF.")
            
        raw_date = month_match.group(0).strip().title()
        date_obj = datetime.strptime(raw_date, "%B %Y")
        
        csv_date = date_obj.strftime("%Y-%m-01")
        print_date = date_obj.strftime("%B %Y")
        
        # 2. Find the Scheduled Passenger Arrivals
        lines = text.split('\n')
        in_scheduled_carriers = False
        
        for line in lines:
            if "Scheduled Carriers" in line:
                in_scheduled_carriers = True
                
            elif in_scheduled_carriers and "Arrivals" in line:
                numbers = re.findall(r'[\d,]+', line)
                clean_numbers = [n.replace(',', '') for n in numbers if n.replace(',', '').isdigit()]
                
                if len(clean_numbers) >= 3:
                    total_arrivals = clean_numbers[2]
                    return csv_date, print_date, total_arrivals
                    
    raise Exception("Scheduled Arrivals data not found in PDF.")

if __name__ == "__main__":
    print("Scraping webpage for VOAT PDF links...")
    links = get_pdf_links("la_webpage.html")
    print(f"Found {len(links)} VOAT reports to process.\n")
    
    output_dir = "Data/Processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scraped_arrivals.csv")
    
    results = []
    
    # If you want to test on just a few first, you can use links[:5]
    # Currently set to process ALL links found in the tables
    for url in links:
        try:
            csv_date, print_date, total_traffic = extract_arrivals_from_pdf(url)
            print(f"-> Total Scheduled Arrivals for {print_date}: {total_traffic}")
            results.append([csv_date, total_traffic])
        except Exception as e:
            # Added a fallback print to show the failing URL since older links don't have filenames
            print(f"-> Failed to process {url}: {e}")
            
    if results:
        print(f"\nSaving data to {output_file}...")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Total Arrivals']) 
            writer.writerows(results)                   
        print("Done!")