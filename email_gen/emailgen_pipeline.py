#!/usr/bin/env python3
import argparse
import json
import os
from dotenv import load_dotenv
from groq import Groq
from email_gen.vector_embedding.retrieve_email import extract_company_text, retrieve_similar_email

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("❌ Missing GROQ_API_KEY in .env")

client = Groq(api_key=API_KEY)

def generate_email(
    company_data_path: str,
    embeddings_path: str,
    company_name: str,
    tone: str = None,
    focus: str = None,
    additional_context: str = None
) -> dict:
    # derive slug for nested JSON lookup
    slug = company_name.strip().lower().replace(" ", "-")

    with open(company_data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # support both { slug: {...} } and single-object formats
    if isinstance(raw, dict) and slug in raw:
        company_data = raw[slug]
    elif isinstance(raw, dict) and "company_name" in raw:
        company_data = raw
    else:
        raise ValueError(f"❌ Company '{company_name}' not found in {company_data_path}")

    source_text   = extract_company_text(company_data)
    similar_email = retrieve_similar_email(source_text, embeddings_path)

    user_prompt = f"""
You are given three blocks of text below. Use them to adapt the email as instructed.

===
[COMPANY NAME]
{company_name}
===
[COMPANY INFO]
{source_text}
===
[TEMPLATE EMAIL]
{similar_email}
===
[TASK]
Adapt the TEMPLATE EMAIL to target the company described in COMPANY NAME and COMPANY INFO.

Incorporate the following guidelines into the adapted email:
- Tone: {tone or 'As per template'}
- Focus: {focus or 'As per template'}
- Additional Context: {additional_context or 'None'}

Rules:
- DO NOT write a new email.
- KEEP the EXACT SAME structure, paragraph breaks, tone, and flow.
- Only swap out company-specific parts using COMPANY INFO and use the company name "{company_name}" in all places.
- NEVER guess company names based on the description.
- NEVER include "Here is the adapted email:"
- NEVER include placeholders like [Your Company].
- Maintain same paragraph count.
- DO NOT add or remove sentences.
- DO NOT add praise or extra opinions.
- Only one final call to action like:
  - "Would you be up for a conversation next week?"
  - "I'd love to hear more about [company/topic]. Would you be free for a quick chat?"
- Output ONLY the subject line and email body.
- Generate a unique and engaging subject line that does NOT start with “Exploring” or similar generic verbs
- Use company-specific context to make the subject relevant

Output format:
Subject: [your subject line]

[email body only, no header or explanation]
""".strip()

    system_prompt = """You are an expert email writer. Adapt the provided TEMPLATE EMAIL using the COMPANY INFO and COMPANY NAME.

Rules:
- Match the template exactly in structure and tone.
- Only change company-specific details.
- Do not invent or generalize anything.
- No placeholders like [Your Company]
- Never include introductory phrases like "Here is the adapted email:"
- End with only ONE call-to-action from the approved list.
- Subject lines must NOT begin with vague verbs like “Exploring”, “Discovering”, “Learning about”, “Understanding”, etc.
- Subject lines should be relevant, natural, and specific to the company’s work or industry.
- Vary the subject style to avoid repetition. Good examples:
   - “How rcg advertising and media elevates brand impact”
   - “Let’s talk brand strategy and media agility”
   - “Helping brands thrive across traditional and digital platforms”

Output only the subject and the email body. No extra content.
""".strip()

    # call the LLM
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.2,
        top_p=0.85
    )
    result = response.choices[0].message.content.strip()

    # cleanup
    if result.lower().startswith("here is the adapted email"):
        result = result.split("\n", 1)[1].strip()
    result = result.replace("[Your Company]", "").strip()

    os.makedirs("data", exist_ok=True)
    with open("generated_data/final_email.txt", "w", encoding="utf-8") as f:
        f.write(result)

    lines = result.splitlines()
    subject = ""
    body    = ""
    if lines and lines[0].lower().startswith("subject:"):
        subject = lines[0].split(":",1)[1].strip()
        body    = "\n".join(lines[1:]).strip()
    else:
        body = result

    return {"company": company_name, "subject": subject, "email": body}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adapt a template email using scraped company_data.json"
    )
    parser.add_argument(
        "-d", "--data",
        dest="company_data_path",
        default="generated_data/company_data.json",
        help="Path to the scraped JSON file"
    )
    parser.add_argument(
        "-e", "--embeddings",
        dest="embeddings_path",
        default="generated_data/email_embeddings.json",
        help="Path to the email embeddings JSON"
    )
    parser.add_argument(
        "-c", "--company",
        dest="company_name",
        default=None,
        help="Company name (slug) to target; falls back to top-level company_name in JSON"
    )
    parser.add_argument("--tone", default=None, help="Desired tone for the email")
    parser.add_argument("--focus", default=None, help="Desired focus for the email")
    parser.add_argument(
        "--additional_context",
        default=None,
        help="Additional context to weave into the email"
    )

    args = parser.parse_args()

    if not args.company_name:
        with open(args.company_data_path, "r", encoding="utf-8") as f:
            top = json.load(f)
        if isinstance(top, dict) and "company_name" in top:
            args.company_name = top["company_name"]
        else:
            parser.error("No --company provided and JSON lacks top-level 'company_name'")

    result = generate_email(
        company_data_path=args.company_data_path,
        embeddings_path=args.embeddings_path,
        company_name=args.company_name,
        tone=args.tone,
        focus=args.focus,
        additional_context=args.additional_context
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
