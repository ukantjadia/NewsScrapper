import os
import json
import re
import warnings
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup, GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

import feedparser
import wikipedia
import wptools
import yfinance as yf
import requests
from textblob import TextBlob
from dotenv import load_dotenv

from newspaper import Article
from boilerpy3 import extractors
boiler_extractor = extractors.ArticleExtractor()

from openai import OpenAI

# ─── Configuration ──────────────────────────────────────────────────────────────
MAX_TOKENS = 1024

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY in your .env file")

router_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def llm_chat(prompt: str) -> Optional[str]:
    try:
        resp = router_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system",  "content": "You are a helpful assistant."},
                {"role": "user",    "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            stream=False
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ OpenRouter call failed ({e}).")
        return None

def tag_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    return {"polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity}

def get_homepage_info(website: str) -> dict:
    try:
        r = requests.get(website, timeout=10)
        r.raise_for_status()
    except:
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    meta = soup.find("meta", attrs={"name": "description"})
    meta_desc = meta["content"].strip() if meta and meta.get("content") else None
    first_h1  = soup.find("h1").get_text().strip() if soup.find("h1") else None
    first_p   = soup.find("p").get_text().strip() if soup.find("p") else None

    headers = []
    for lvl in range(1,7):
        for tag in soup.find_all(f"h{lvl}"):
            txt = tag.get_text().strip()
            if txt:
                headers.append({"tag": f"h{lvl}", "text": txt})

    return {
        "meta_description":  meta_desc,
        "first_h1":          first_h1,
        "homepage_snippet":  first_p,
        "headers":           headers
    }

def get_wikipedia_description(company: str) -> Optional[str]:
    try:
        page = wikipedia.page(company, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        choice = next((opt for opt in e.options if company.lower() in opt.lower()),
                      e.options[0])
        try:
            page = wikipedia.page(choice, auto_suggest=False)
        except:
            return None
    except:
        return None
    para = page.content.split("\n\n",1)[0]
    return re.sub(r"\[\d+\]", "", para).strip()

def get_wikipedia_infobox(company: str) -> dict:
    try:
        wp = wptools.page(company, silent=True)
        wp.get_parse()
        return wp.data.get("infobox",{}) or {}
    except:
        return {}

def fetch_clean_text(url: str) -> str:
    try:
        art = Article(url)
        art.download(); art.parse()
        return art.text or ""
    except:
        pass
    try:
        html = requests.get(url, timeout=10).text
        return boiler_extractor.get_content(html)
    except:
        return ""

def llm_summarize_article(company: str, title: str, text: str) -> str:
    prompt = f"""
You are an expert news analyst for {company}.
Read the following article text and produce a 4–5 sentence summary,
focusing only on aspects relevant to {company}'s business, strategy, or products.

Title: {title}

Article text:
\"\"\"{text}\"\"\"
"""
    out = llm_chat(prompt)
    return out or ""

def get_google_news_articles(company: str,
                             query: str,
                             max_items: int = 5) -> List[dict]:
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(rss_url)
    results = []
    for entry in feed.entries[:max_items]:
        text    = fetch_clean_text(entry.link)
        summary = llm_summarize_article(company, entry.title, text) if text else ""
        results.append({
            "title":     entry.title,
            "link":      entry.link,
            "published": entry.get("published"),
            "summary":   summary,
            "sentiment": tag_sentiment(summary)
        })
    return results

def filter_recent_articles(arts: List[dict], days: int = 30) -> List[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for a in arts:
        try:
            dt = parsedate_to_datetime(a["published"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                out.append(a)
        except:
            continue
    return out

def extract_keywords(company: str, context: str,
                     max_keywords: int = 5) -> List[str]:
    prompt = f"""
Extract up to {max_keywords} key phrases (2–4 words each)
describing {company}'s core business:
\"\"\"{context}\"\"\"
Return ONLY a JSON array of strings.
"""
    raw = llm_chat(prompt)
    try:
        return json.loads(re.sub(r"^```[a-z]*|```$", "", raw, flags=re.I))
    except:
        return []

def get_stock_info(infobox: dict) -> dict:
    ticker = None
    for key in ("ticker","traded_as","stock_symbol"):
        val = infobox.get(key)
        if isinstance(val,str):
            m = re.search(r"[A-Za-z.]+$", val)
            if m:
                ticker = m.group(0); break
    if not ticker:
        return {}
    try:
        info = yf.Ticker(ticker).info
        return {
            "current_price":  info.get("regularMarketPrice"),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "beta":           info.get("beta"),
        }
    except:
        return {}

def build_full_profile(company: str,
                       website: str,
                       max_news: int = 5) -> dict:
    """Run the full company_profile pipeline headlessly."""
    homepage     = get_homepage_info(website)
    wiki_summary = get_wikipedia_description(company)
    wiki_info    = get_wikipedia_infobox(company)
    context      = "\n\n".join(filter(None,
                        [homepage.get("homepage_snippet"), wiki_summary]))
    keywords     = extract_keywords(company, context)[:2]
    query        = company if not keywords else (
                   f"{company} AND ({keywords[0]} OR {keywords[1]})")
    raw_news     = get_google_news_articles(company, query, max_items=max_news)
    recent_news  = filter_recent_articles(raw_news)
    stock        = get_stock_info(wiki_info)

    profile = {
        "homepage":    homepage,
        "wikipedia":   {"summary": wiki_summary,
                        "infobox": wiki_info},
        "recent_news": recent_news
    }
    if stock:
        profile["stock_info"] = stock

    return profile
