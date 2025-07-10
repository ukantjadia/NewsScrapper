# extract_features.py
def extract_features(company_data: dict) -> str:
    parts = []

    if "description" in company_data:
        parts.append(f"Description: {company_data['description']}")
    if "company_overview" in company_data:
        parts.append(f"Overview: {company_data['company_overview']}")
    if "news" in company_data and isinstance(company_data["news"], str):
        parts.append(f"Recent News: {company_data['news'][:400]}")  # Limit length
    if "industry_categories" in company_data:
        parts.append("Industries: " + ", ".join(company_data["industry_categories"]))
    if "founder_info" in company_data:
        for name, info in company_data["founder_info"].items():
            summary = info.get("wikipedia_summary", "")
            parts.append(f"Founder: {name} - {summary}")

    return "\n".join(parts)
