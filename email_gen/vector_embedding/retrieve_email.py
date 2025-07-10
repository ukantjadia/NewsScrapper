from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_company_text(data: dict) -> str:
    """
    Build a text blob from:
      • description
      • company_overview (if any)
      • industries
      • website_summary
      • all entries in `news` array
    """
    desc      = data.get("description", "")
    overview  = data.get("company_overview", "")
    industries= ", ".join(data.get("industry_categories", []))
    website   = data.get("website_summary", "")

    # Format each news entry
    lines = []
    for item in data.get("news", []):
        date    = item.get("news_date", "")
        title   = item.get("title", "")
        summary = item.get("summary") or ""
        lines.append(f"{date} — {title}: {summary}")

    news_text = "\n".join(lines)

    return (
        f"{desc}\n\n"
        f"{overview}\n\n"
        f"Industries: {industries}\n\n"
        f"Website Summary: {website}\n\n"
        f"Recent News:\n{news_text}"
    )

def retrieve_similar_email(company_text: str, embeddings_file: str) -> str:
    """
    Given the combined company_text, find the best match
    from your precomputed email embeddings.
    """
    query_emb = model.encode(company_text)
    with open(embeddings_file, "r", encoding="utf-8") as f:
        embedded = json.load(f)

    best_score, best_email = -1, None
    for entry in embedded:
        emb   = np.array(entry["embedding"], dtype=np.float32)
        score = util.cos_sim(query_emb, emb).item()
        if score > best_score:
            best_score, best_email = score, entry["email"]

    if best_email is None:
        raise RuntimeError("No similar email found.")
    return best_email
