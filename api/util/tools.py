from bs4 import BeautifulSoup
import requests
import re
import sys
import io
import os
import csv
import base64
import base64
from io import BytesIO
from PIL import Image

from PIL import Image
import base64

def image_to_base64(image_path: str) -> str:
    with Image.open(image_path) as img:
        img_format = img.format.lower()
        with open(image_path, "rb") as img_file:
            encoded_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/{img_format};base64,{encoded_str}"



def scrape_html(x: str)->str:
    url=x
    tag='p'
    """
    Scrape the HTML content of a given tag from a URL.
    """
    print(url,tag)
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    elements = soup.find_all(tag)
    
    return ' '.join([element.get_text() for element in elements])

def scrape_table_html(url: str, max_tokens: int = 30000) -> str:
    """
    Scrape relevant HTML tables from a URL, optimized for LLM/ReAct agent usage.
    Only tables with meaningful data (at least 2 rows and 2 columns) are included.
    Returns tables in CSV format (comma-separated, quoted as needed).
    Token budget is respected to avoid overflows.
    """

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all("table")
    csv_tables = []
    total_tokens = 0

    def count_tokens(text):
        # Approximate token count: 1 token ≈ 4 chars (for English text)
        return max(1, len(text) // 4)

    for table in tables:
        rows = table.find_all('tr')
        if len(rows) < 2:
            continue
        first_row_cells = rows[0].find_all(['th', 'td'])
        if len(first_row_cells) < 2:
            continue

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            cells = row.find_all(['th', 'td'])
            cell_text = []
            for cell in cells:
                text = cell.get_text(separator=' ', strip=True)
                text = re.sub(r'\[\w+\]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                cell_text.append(text)
            if cell_text:
                row_text = ','.join(cell_text)
                row_tokens = count_tokens(row_text)
                if total_tokens + row_tokens > max_tokens:
                    break
                writer.writerow(cell_text)
                total_tokens += row_tokens
        csv_content = csv_buffer.getvalue().strip()
        if csv_content:
            table_tokens = count_tokens(csv_content)
            if total_tokens + table_tokens > max_tokens:
                break
            csv_tables.append(csv_content)
            total_tokens += table_tokens
        if total_tokens >= max_tokens:
            break

    if not csv_tables:
        return "No relevant tables found or token budget exceeded."
    return '\n\n---\n\n'.join(csv_tables)

def read_csv(file_name: str) -> str:
    """
    Read a CSV file and return its content as a string.
    """
    try:
        with open(os.getcwd()+"/"+file_name, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File {file_name} not found."
    except Exception as e:
        return f"An error occurred while reading the file: {str(e)}"

def read_json(file_name: str) -> str:
    """
    Read a JSON file and return its content as a string.
    """
    try:
        with open(os.getcwd()+"/"+file_name, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return f"File {file_name} not found."
    except Exception as e:
        return f"An error occurred while reading the file: {str(e)}"


