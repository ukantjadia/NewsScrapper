import time
import csv
import json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn

# Your combined pipeline
from scrapers.scraper_pipeline import scrape_company_info
from email_gen.emailgen_pipeline import generate_email
from email_gen.linkedin_message_gen import generate_linkedin_message

# ========== CONFIG ==========

CSV_FILE = "email_log.csv"
JSON_FILE = "generated_data/company_data.json"
EMBEDDINGS_PATH = "generated_data/email_embeddings.json"

# ========== FASTAPI APP ==========

app = FastAPI()
latest_email = {"company": "", "subject": "", "email": ""}

# ========== CSV LOGGER ==========

def append_to_csv(entry: dict):
    fieldnames = ["company", "subject", "email"]
    try:
        with open(CSV_FILE, "a", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(entry)
    except Exception as e:
        print(f"❌ Failed to write to CSV: {e}")

# ========== FASTAPI ENDPOINTS ==========

class CompanyRequest(BaseModel):
    company_name: str
    homepage_url: str
    tone: Optional[str] = Field(None, description="Tone of the message (e.g., Professional, Friendly)")
    focus: Optional[str] = Field(None, description="Focus of the message (e.g., Partnership, Collaboration)")
    additional_context: Optional[str] = Field(None, description="Any extra context to include")

@app.post("/scrape")
def scrape_and_generate(request: CompanyRequest):
    global latest_email
    start = time.time()

    # 1. Run full data scraping pipeline (Crunchbase + founders + news + website summary)
    scrape_company_info(request.company_name, request.homepage_url)

    # 1.5. Ensure company_name is present in the JSON
    with open(JSON_FILE, "r+", encoding="utf-8") as f:
        data = json.load(f)
        if "company_name" not in data or not data["company_name"]:
            data["company_name"] = request.company_name
            f.seek(0)
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.truncate()

    # 2. Generate email using updated company_data.json
    email = generate_email(
        JSON_FILE,
        EMBEDDINGS_PATH,
        company_name = request.company_name, # this helps do the key value search in the company_data.json file
        tone=request.tone,
        focus=request.focus,
        additional_context=request.additional_context
    )
    append_to_csv(email)
    latest_email = email

    # 3. Generate LinkedIn message using updated company_data.json
    linkedin_message = generate_linkedin_message(
        JSON_FILE,
        tone=request.tone,
        focus=request.focus,
        additional_context=request.additional_context
    )

    print(f"✅ API request completed in {time.time() - start:.2f}s")
    return JSONResponse(content={
        "company": email["company"],
        "subject": email["subject"],
        "email": email["email"],
        "linkedin_message": linkedin_message
    })

@app.get("/email")
def get_latest_email():
    return latest_email

# ========== MAIN RUNNER ==========

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
