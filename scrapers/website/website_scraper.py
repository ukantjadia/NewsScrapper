import os
import requests
from bs4 import BeautifulSoup
from collections import OrderedDict
from groq import Groq
import urllib3
from dotenv import load_dotenv

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file
load_dotenv()
MAX_SECTIONS = 10  # Limit to top 10 sections

def load_api_key():
    return os.getenv("GROQ_API_KEY")

def try_fetch_url(url: str) -> str:
    """
    Try HTTPS with verify=False first, then fallback to HTTP if SSL error occurs.
    Returns HTML content if successful, raises otherwise.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, verify=False, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.SSLError:
        print("⚠️ SSL error. Trying HTTP fallback...")
        url = url.replace("https://", "http://")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        return response.text

def scrape_homepage_sections(url: str) -> dict:
    """
    Scrapes visible, meaningful homepage sections using headers and structure.
    Restricts to top MAX_SECTIONS sections for speed.
    """
    try:
        html = try_fetch_url(url)
    except Exception as e:
        print(f"❌ Failed to scrape website: {e}")
        return {}

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "svg", "button"]):
        tag.decompose()

    main = soup.find("main") or soup.body
    sections = OrderedDict()
    count = 1

    for section in main.find_all(["section", "div"], recursive=True):
        if len(sections) >= MAX_SECTIONS:
            break

        class_str = " ".join(section.get("class", []))
        if any(skip in class_str.lower() for skip in ["footer", "chat", "nav", "header"]):
            continue

        text = section.get_text(separator="\n", strip=True)
        if not text or len(text) < 100:
            continue

        header = section.find(["h1", "h2", "h3"])
        title = header.get_text(strip=True) if header else f"Section {count}"
        while title in sections:
            count += 1
            title = f"{title}_{count}"

        sections[title] = text
        count += 1

    return sections

def summarize_full_site_with_groq(sections: dict, max_tokens=2048) -> str:
    """
    Combines all section texts and produces one unified summary.
    """
    if not sections:
        return "No content could be scraped from the website."

    api_key = load_api_key()
    client = Groq(api_key=api_key)

    combined_content = "\n\n".join(sections.values())[:6000]

    prompt = f"""
You are an expert business analyst. Analyze the following combined content scraped from a company's homepage.
Extract the most important information and return only bullet points.

Do not include any introductions like "Here is a summary" or "Below is a summary." Just the bullet points.

Your summary should cover:
- What the company is about
- What products and services they offer
- Key industries or use cases they serve
- Any major platforms or technologies mentioned

Combined Website Content:
{combined_content}
"""

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content.strip()

def compile_summary(summaries: dict) -> str:
    output = ["=== Website Summary by Section ===\n"]
    for header, summary in summaries.items():
        output.append(f"## {header}\n{summary}\n")
    return "\n".join(output)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape a company website and extract structured insights.")
    parser.add_argument("url", help="The URL of the company website to analyze.")
    args = parser.parse_args()

    print("🔍 Scraping and analyzing top homepage sections...")
    sections = scrape_homepage_sections(args.url)
    summary = summarize_full_site_with_groq(sections)
    print("\n📄 Final Summary:\n")
    print(summary)
