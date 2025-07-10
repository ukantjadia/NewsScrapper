import json
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_linkedin_prompt_template() -> str:
    """
    Returns the prompt template for generating a LinkedIn message.
    """
    return """
You are an outreach assistant writing LinkedIn messages to industry professionals.
Your goal is to start a conversation or get a short meeting. Follow these strict rules:

Linkedin Message Requirements:
    1. Start by referencing a specific achievement or news item from the source.
    2. Include an insightful question about their strategy or a challenge they might face.
    3. Conclude with a prompt to discuss strategy or potential next steps.
    4. Tone must be professional and investor-like (not salesy or impressed)

    Content Rules:
    - Focus on what makes the company uniquely interesting for acquisition
    - Ask specific questions about business strategy
    - Discuss future potential rather than past accomplishments
    - NO greeting like 'Dear' or 'Hi Dr. ___'
    - NO long intro
    - NO sign-off or full name
    - DO NOT add any conversational filler or preambles like "Here is the LinkedIn message:". Output only the message itself.
    - Reference a specific project, news, or achievement from their background.
    - Never use: impressed, fascinated, admire, excited, appreciate
    - Never mention: company culture, testimonials, diversity status
    - Keep sentences concise (readable in one breath)
    - LinkedIn has a short word limit — keep it under 60 words.
    - Keep tone friendly and professional — not salesy or too formal.
    - Do NOT use subject lines, "Dear", long intros, or email closings.
    - Never add your contact details, title, or long signature, and no need to add Best Regards at last.
    - NEVER use: "excited", "appreciate", "admire", "partnership", "collaboration", or "diversity"


Output Format:
[Your LinkedIn message only — no greetings or sign-offs]

Generate a LinkedIn message following the above rules.
"""

def extract_company_info_for_linkedin(data: dict) -> str:
    """
    Extracts a concise summary of the company for the LinkedIn prompt.
    """
    website_summary = data.get("website_summary", "")
    description = data.get("description", "")
    overview = data.get("company_overview", "")
    news_summary = data.get("news_summary", "")

    # Prioritize website_summary, then overview, then description
    content = website_summary or overview or description

    # Add a snippet of news if available
    if news_summary:
        content += f"\n\nRecent News Highlight: {news_summary}..."

    return content.strip()

def generate_linkedin_message(
    company_data_path: str,
    tone: str = None,
    focus: str = None,
    additional_context: str = None
) -> str:
    """
    Generates a personalized LinkedIn message based on company data.

    Args:
        company_data_path (str): The path to the company_data.json file.
        tone (str): The desired tone for the LinkedIn message.
        focus (str): The focus of the LinkedIn message.
        additional_context (str): Additional context to include in the LinkedIn message.

    Returns:
        str: The generated LinkedIn message.
    """
    try:
        with open(company_data_path, "r", encoding="utf-8") as f:
            company_data = json.load(f)
    except FileNotFoundError:
        return "Error: company_data.json not found."
    except json.JSONDecodeError:
        return "Error: Could not decode company_data.json. The file might be empty or malformed."

    # print(company_data)
    company_name = company_data.get("company_name", "the company")
    company_info = extract_company_info_for_linkedin(company_data)
    prompt_template = get_linkedin_prompt_template()

    # Add customization to the prompt
    customization_prompt = f"""
COMPANY TO TARGET: {company_name}

COMPANY INFORMATION:
{company_info}
"""
    if tone:
        customization_prompt += f"\nTONE: Must be {tone}"
    if focus:
        customization_prompt += f"\nFOCUS: The message focus should be on {focus}"
    if additional_context:
        customization_prompt += f"\nADDITIONAL CONTEXT: Incorporate this key point: {additional_context}"

    # Format the final prompt
    final_prompt = prompt_template + "\n" + customization_prompt

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Using a smaller model for a short message is efficient
            messages=[
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.7,
            top_p=0.9
        )
        message = response.choices[0].message.content.strip()
        print(f"linkedin messages {message}")
        return message
    except Exception as e:
        print(f"❌ Error generating LinkedIn message with Groq: {e}")
        return "Error: Failed to generate message due to an API error."

if __name__ == "__main__":
    # Example usage:
    # Assumes 'generated_data/company_data.json' exists from a previous step.
    json_path = os.path.join("data", "company_data.json")

    # Create a dummy json if it doesn't exist for testing
    if not os.path.exists(json_path):
        os.makedirs("data", exist_ok=True)
        dummy_data = {
          "company_name": "InnovateTech",
          "description": "InnovateTech is a leading provider of AI-driven solutions for the tech industry.",
          "company_overview": "At InnovateTech, we build cutting-edge software that helps businesses optimize their workflow. Our latest product, 'OptimizeAI', just won a major industry award for innovation.",
          "news_summary": "InnovateTech was recently featured in TechCrunch for its groundbreaking work in machine learning and its successful Series B funding round."
        }
        with open(json_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        print(f"Created dummy data file at {json_path}")

    print(json_path)
    print("🚀 Generating LinkedIn message...")
    linkedin_message = generate_linkedin_message(json_path)
    print("\n--- Generated LinkedIn Message ---\n")
    print(linkedin_message)
    print("\n--------------------------------\n")