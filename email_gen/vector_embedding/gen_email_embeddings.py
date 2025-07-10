from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("generated_data/emails.json", "r", encoding="utf-8") as f:
    emails = json.load(f)

email_embeddings = []
for idx, entry in enumerate(emails):
    email_text = entry["email"]
    vector = model.encode(email_text)
    email_embeddings.append({
        "id": idx,
        "email": email_text,
        "embedding": vector.tolist()
    })

with open("generated_data/email_embeddings.json", "w") as f:
    json.dump(email_embeddings, f, indent=2)
