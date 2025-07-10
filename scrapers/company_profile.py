import os
import re
import json
import warnings
from typing import Optional, List, Dict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import requests
from bs4 import BeautifulSoup, GuessedAtParserWarning
import feedparser
import wikipedia
import wptools
import yfinance as yf
from textblob import TextBlob

from newspaper import Article
from boilerpy3 import extractors
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Suppress warnings
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# ─── Load OpenRouter API key ───────────────────────────────────────────────────
load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Please set OPENROUTER_API_KEY in your .env file")

router_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

MAX_TOKENS = 1024
boiler_extractor = extractors.ArticleExtractor()


def llm_chat(prompt: str) -> Optional[str]:
    """
    Send prompt to OpenRouter (Gemini model) and return the assistant's reply.
    """
    try:
        resp = router_client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system",  "content": "You are a helpful assistant."},
                {"role": "user",    "content": prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.5,
            stream=False
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ OpenRouter call failed ({e}).")
        return None


def tag_sentiment(text: str) -> Dict:
    blob = TextBlob(text)
    return {
        "polarity":     blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }


def get_homepage_info(website: str) -> Dict:
    try:
        r = requests.get(website, timeout=10)
        r.raise_for_status()
    except:
        return {}
    soup = BeautifulSoup(r.text, "lxml")
    meta = soup.find("meta", {"name": "description"})
    return {
        "meta_description": meta["content"].strip() if meta and meta.get("content") else None,
        "first_h1":         (soup.find("h1").get_text().strip() if soup.find("h1") else None),
        "homepage_snippet": (soup.find("p").get_text().strip()  if soup.find("p")  else None),
        "headers": [
            {"tag": tag.name, "text": tag.get_text(strip=True)}
            for lvl in range(1, 7)
            for tag in soup.find_all(f"h{lvl}")
            if tag.get_text(strip=True)
        ]
    }


def get_wikipedia_description(company: str) -> Optional[str]:
    try:
        page = wikipedia.page(company, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        choice = next((o for o in e.options if company.lower() in o.lower()), e.options[0])
        try:
            page = wikipedia.page(choice, auto_suggest=False)
        except:
            return None
    except:
        return None
    para = page.content.split("\n\n", 1)[0]
    return re.sub(r"\[\d+\]", "", para).strip()


def get_wikipedia_infobox(company: str) -> Dict:
    try:
        wp = wptools.page(company, silent=True)
        wp.get_parse()
        return wp.data.get("infobox", {}) or {}
    except:
        return {}


def extract_keywords(company: str, context: str, max_keywords: int = 5) -> List[str]:
    prompt = (
        f"Extract up to {max_keywords} key phrases (2–4 words each) "
        f"describing {company}'s core business:\n\n{context}\n\n"
        "Return ONLY a JSON array of strings."
    )
    raw = llm_chat(prompt) or ""
    try:
        return json.loads(re.sub(r"^```[a-z]*|```$", "", raw, flags=re.I))
    except:
        return []


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
    prompt = (
        f"You are an expert news analyst for {company}. "
        f"Read the following article text and produce a 4–5 sentence summary:\n\n"
        f"Title: {title}\n\n{text}"
    )
    return llm_chat(prompt) or ""


def get_google_news_articles(company: str, query: str, max_items: int = 5) -> List[Dict]:
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    )
    feed = feedparser.parse(rss_url)
    out = []
    for entry in feed.entries[:max_items]:
        txt = fetch_clean_text(entry.link)
        out.append({
            "title":     entry.title,
            "link":      entry.link,
            "published": entry.get("published"),
            "summary":   llm_summarize_article(company, entry.title, txt) if txt else "",
            "sentiment": tag_sentiment(txt)
        })
    return out


def filter_recent_articles(arts: List[Dict], days: int = 30) -> List[Dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    ret = []
    for a in arts:
        try:
            dt = parsedate_to_datetime(a["published"])
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt >= cutoff:
                ret.append(a)
        except:
            continue
    return ret


def get_stock_info(infobox: Dict) -> Dict:
    for key in ("ticker", "traded_as", "stock_symbol"):
        val = infobox.get(key)
        if isinstance(val, str):
            m = re.search(r"[A-Za-z.]+$", val)
            if m:
                info = yf.Ticker(m.group(0)).info
                return {
                    "current_price":  info.get("regularMarketPrice"),
                    "market_cap":     info.get("marketCap"),
                    "pe_ratio":       info.get("trailingPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "beta":           info.get("beta")
                }
    return {}


def get_realtime_news(company: str) -> List[Dict]:
    url = "https://real-time-news-data.p.rapidapi.com/search"
    qs  = {"query": company, "limit": "5", "country": "US", "lang": "en"}
    hdr = {
        "x-rapidapi-host": "real-time-news-data.p.rapidapi.com",
        "x-rapidapi-key":  "bf8b2ea17emsh82563bd04a647e6p1fc3e5jsn7086c1a7b4a0"
    }
    try:
        r = requests.get(url, headers=hdr, params=qs, timeout=10)
        r.raise_for_status()
        items = r.json().get("data", [])
    except Exception as e:
        print(f"⚠️ Real-time news fetch error: {e}")
        return []
    return [{
        "title":     it.get("title"),
        "link":      it.get("link") or it.get("source_url"),
        "published": it.get("published_datetime_utc"),
        "source":    it.get("source_name")
    } for it in items]


def get_section_news(company: str) -> List[Dict]:
    token = "<YOUR_SECTION_TOKEN>"
    url = "https://real-time-news-data.p.rapidapi.com/topic-news-by-section"
    qs  = {"topic": company, "section": token, "limit": "5", "country": "US", "lang": "en"}
    hdr = {
        "x-rapidapi-host": "real-time-news-data.p.rapidapi.com",
        "x-rapidapi-key":  "0a1624f494mshbc423fe3e4bda47p12a33bjsnccf418710fb4"
    }
    try:
        r = requests.get(url, headers=hdr, params=qs, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"⚠️ Section news fetch error: {e}")
        return []
    return [{
        "title":     it.get("title"),
        "link":      it.get("link") or it.get("source_url"),
        "published": it.get("published_datetime_utc"),
        "source":    it.get("source_name")
    } for it in data]


def build_full_profile(company: str, website: str, max_news: int = 5) -> Dict:
    homepage      = get_homepage_info(website)
    wiki_summary  = get_wikipedia_description(company)
    wiki_infobox  = get_wikipedia_infobox(company)

    context       = "\n\n".join(filter(None, [homepage.get("homepage_snippet"), wiki_summary]))
    keywords      = extract_keywords(company, context)[:2]
    query         = company if not keywords else f"{company} AND ({keywords[0]} OR {keywords[1]})"

    recent_news   = filter_recent_articles(get_google_news_articles(company, query, max_items=max_news))
    realtime_news = get_realtime_news(company)
    section_news  = get_section_news(company)
    stock_info    = get_stock_info(wiki_infobox)

    profile = {
        "homepage":      homepage,
        "wikipedia":     {"summary": wiki_summary, "infobox": wiki_infobox},
        "recent_news":   recent_news,
        "realtime_news": realtime_news,
        "section_news":  section_news
    }
    if stock_info:
        profile["stock_info"] = stock_info

    return profile
