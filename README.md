# AI-Powered HR Assessment Recommender 🎯

An AI-powered agent designed to guide recruiters from vague hiring intents to a grounded shortlist of talent assessments using **Groq (Llama 3.3)** and **RAG (TF-IDF)** on the TalentAxis catalog.

## 🚀 Live Demo & Links
- **Live API:** [https://talentaxis-assessment-recommender.onrender.com/](https://talentaxis-assessment-recommender.onrender.com/)
- **Interactive UI:** Open `frontend/index.html` locally to access the Recruiter Dashboard.
- **API Docs:** [Swagger UI](https://talentaxis-assessment-recommender.onrender.com/docs)

---

## 📸 Project Screenshots

### 📂 API Endpoints Overview
![API Endpoints](./api_endpoints.png)

### 📋 Response Schema
![API Schema](./api_schema.png)

### ⚡ Live API Interaction
![Live Test](./live_test.png)

---

## ✨ Key Features
- **Intelligent Clarification:** Bot asks for role details, seniority, and skills before recommending.
- **RAG Powered:** Uses a TF-IDF vectorizer to search the generic HR assessment catalog without hallucinations.
- **Case-Insensitive API:** Handles `user`, `USER`, or `Human` roles seamlessly.
- **Memory Optimized:** Specifically designed to run within the 512MB RAM limit of free tier hosting.
- **Premium UI:** Side-by-side chat and recommendation engine for better recruiter experience.

## 🛠️ Tech Stack
- **Backend:** FastAPI (Python)
- **LLM:** Groq API (Llama 3.3 70B)
- **Vector Search:** Scikit-learn (TF-IDF)
- **Frontend:** Vanilla JS + Tailwind CSS
- **Deployment:** Render

## 📝 Setup & Usage
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set environment variable: `export GROQ_API_KEY=your_key_here`.
4. Run locally: `uvicorn main:app --reload`.
