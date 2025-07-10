#!/usr/bin/env python3

import os
import sys
import json
import argparse
from datetime import datetime, timezone

# ensure project root is on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from scrapers.website.website_scraper import scrape_homepage_sections, summarize_full_site_with_groq
from scrapers.company_profile import (
    get_google_news_articles,
    filter_recent_articles,
    get_realtime_news,
    fetch_clean_text
)
from email_gen.emailgen_pipeline import generate_email

def scrape_company_info(company_name: str, company_url: str, max_news: int) -> dict:
    """
    Returns JSON in the specified minimal format:
    {
      "company_name": "<slug>",
      "company_url": "<url>",
      "description": "<homepage summary>",
      "scraped_date": "<ISO timestamp>",
      "news": [ { title, scrapper_source, url, summary, content, news_date }, … ]
    }
    """
    # 1) Website summary
    sections    = scrape_homepage_sections(company_url)
    description = summarize_full_site_with_groq(sections)

    # 2) Timestamp
    scraped_date = datetime.now(timezone.utc).isoformat()

    # 3) Google RSS news (last 30 days)
    raw_google = get_google_news_articles(company_name, company_name, max_items=max_news)
    recent     = filter_recent_articles(raw_google, days=30)
    google_news = []
    for item in recent:
        url = item.get("link", "")
        google_news.append({
            "title":           item.get("title", ""),
            "scrapper_source": "Google RSS",
            "url":             url,
            "summary":         item.get("summary", ""),
            "content":         fetch_clean_text(url),
            "news_date":       item.get("published", "")
        })

    # 4) RapidAPI real-time news
    raw_rapid = get_realtime_news(company_name)
    rapid_news = []
    for item in raw_rapid:
        url = item.get("link", "")
        rapid_news.append({
            "title":           item.get("title", ""),
            "scrapper_source": "RapidAPI",
            "url":             url,
            "summary":         "",
            "content":         "",
            "news_date":       item.get("published", "")
        })

    # 5) Combine
    news = google_news + rapid_news

    return {
        "company_name":  company_name.strip().lower().replace(" ", "-"),
        "company_url":   company_url,
        "description":   description,
        "scraped_date":  scraped_date,
        "news":          news
    }

def main():
    parser = argparse.ArgumentParser(description="Scrape minimal JSON + generate email")
    parser.add_argument("company_name", help="Company name (used as slug)")
    parser.add_argument(
        "-u", "--url",
        dest="company_url",
        required=True,
        help="Company homepage URL"
    )
    parser.add_argument(
        "-n", "--max-news",
        dest="max_news",
        type=int,
        default=5,
        help="Max items per news source"
    )
    args = parser.parse_args()

    # prompt for tone/focus/context
    tone    = input("Enter email tone (e.g. Friendly): ").strip() or None
    focus   = input("Enter email focus (e.g. Partnership): ").strip() or None
    context = input("Any additional context? (leave blank if none): ").strip() or None

    # run scrape
    os.makedirs("data", exist_ok=True)
    blob = scrape_company_info(args.company_name, args.company_url, args.max_news)

    # write new JSON format
    out_path = "generated_data/company_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)

    print("\n=== Scraped JSON ===")
    print(json.dumps(blob, indent=2))

    # generate email
    email = generate_email(
        out_path,
        "generated_data/email_embeddings.json",
        args.company_name,
        tone=tone,
        focus=focus,
        additional_context=context
    )

    print("\n=== Generated Email ===")
    print(f"Subject: {email['subject']}\n")
    print(email["email"])


if __name__ == "__main__":
    main()
