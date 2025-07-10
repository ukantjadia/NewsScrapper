# 🧠 Cold Email Generator

An end-to-end AI-powered tool to automatically generate personalized cold outreach emails using real-time company data from multiple sources.

---

## 🚀 Features

* **📊 Automated Company Profiling**

  * Scrapes data from **Crunchbase**, **company websites**, **news articles (Google news RSS and RapidAPI news articles), and Wikipedia**.
  * Summarizes company offerings using **LLMs** (Groq + OpenRouter).
  * Builds a consistent, structured company profile JSON.

* **✉️ Email Generation**

  * Retrieves the most relevant past email via **sentence embeddings**.
  * Adapts it using **Gemini (via OpenRouter)** to match your tone, focus, and context.
  * Outputs a subject line and three-sentence cold email.

* **⚡ Fast & Scalable**

  * Full scrape + email generation takes under **10 seconds** per company.

---

## 🛠 Requirements

* Python 3.9+
* `.env` with:

  ```env
  GROQ_API_KEY=your_groq_key
  OPENROUTER_API_KEY=your_openrouter_key
  RAPIDAPI_KEY=your_rapidapi_key
  ```
