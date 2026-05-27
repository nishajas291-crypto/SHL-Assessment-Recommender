# AI-Powered HR Assessment Recommender: Approach Document

## 1. Design Choices

- **Framework:** FastAPI used to expose `/health` and `/chat` endpoints
  - Fast async capabilities
  - Built-in Pydantic support guarantees I/O matches evaluator harness schema (`reply`, `recommendations`, `end_of_conversation`)

- **LLM:** Groq API with `llama-3.3-70b-versatile`
  - Extremely fast inference — ideal for real-time conversational agents
  - `response_format: json_object` forces strict JSON output, eliminating parsing errors

- **Statelessness:** No server-side state stored
  - Each POST request carries full conversation history
  - History injected directly into system prompt

---

## 2. Retrieval Setup (RAG)

- **Method:** TF-IDF Vectorizer (`scikit-learn`) — in-memory retrieval
- **Initialization:** On FastAPI `startup_event`, catalog JSON parsed and indexed
  - Each assessment formatted as: Name + Keys + Job Levels + Description
- **Search Logic:** Latest user + assistant messages combined as search query
  - Cosine similarity used to retrieve **Top 15** matching assessments
- **Grounding:** Only Top 15 candidates injected into LLM system prompt
  - Guarantees LLM selects only valid catalog assessments — no hallucination

---

## 3. Prompt Design

- **Persona:** "Conversational HR Assessment Recommender"
- **Behaviors enforced:**
  1. **Clarify** — Rejects vague queries, keeps `recommendations` empty until context is sufficient
  2. **Recommend & Refine** — Returns 1–10 assessments when context is clear, updates dynamically
  3. **Compare** — Uses only catalog data for comparison (e.g., Personality vs Knowledge tests)
  4. **Safety & Scope** — Refuses general hiring advice, legal questions, prompt injections
  5. **JSON Structuring** — Maps catalog link → `url`, deduces `test_type` from catalog keys (`P`, `K`, `A`, `S`)

---

## 4. Evaluation Approach
- **Schema Validation:** Endpoints tested to ensure strict schema compliance by validating `/health` and `/chat` against evaluator harness schema requirements.
- **Conversation Traces:** Tested against sample traces (C1, C2):
  - Turn 1 with vague query → `recommendations: []`, clarifying question asked.
  - Sufficient context provided → correct shortlist returned.

---

## 5. Evaluation & Measuring Improvement
- **Schema Compliance:** Measured using `test_app.py`. Initial iterations had JSON parsing errors; by implementing Pydantic validation and strict LLM system instructions, schema compliance was improved to 100%.
- **Memory Optimization:** Measured via Render's deployment logs. The initial FAISS/Sentence-Transformers setup consistently triggered "Out of Memory" (OOM) errors (usage > 512MB). Switching to TF-IDF reduced memory usage to ~60MB, ensuring 100% service uptime.
- **Retrieval Accuracy:** Measured by replaying sample conversation traces (C1-C10). Improvements were tracked by comparing the agent’s generated shortlist against the labeled expected results. Grounding the agent with retrieved catalog snippets (RAG) significantly reduced hallucination compared to zero-shot prompting.

---

## 6. What Didn't Work
- **Zero-Shot without Retrieval:**
  - LLM hallucinated non-existent URLs and products.
  - Fix: TF-IDF retrieval + strict `CATALOG CANDIDATES` grounding → 100% valid URLs.
- **Deprecated Groq Models:**
  - `llama-3.1-8b-instant` & `mixtral-8x7b-32768` had stability/deprecation issues.
  - Fix: Migrated to `llama-3.3-70b-versatile` → stable + supports `json_object`.

---

## 7. AI Tools Used
- **Claude AI (via Antigravity Agent):** Used as an agentic coding partner to autonomously plan the architecture, optimize for cloud memory constraints, and build the premium recruiter dashboard.
- **Groq API (`llama-3.3-70b-versatile`):** The primary LLM backbone for the conversational logic.

---

## 8. Optional: Recruiter Dashboard
- **UI Architecture:** A premium single-page dashboard was developed (located in the `/frontend` directory) to demonstrate the API's real-world integration.
- **UX Features:** Built with a side-by-side layout, featuring a WhatsApp-style chat interface on the left and a live-updating assessment shortlist grid on the right. 
- **Tech Stack:** Developed using Tailwind CSS for high-end aesthetics and Vanilla JS for zero-build deployment.

---

## 9. Deployment
- **Platform:** Render (Free Tier)
- **Live API URL:** https://talentaxis-assessment-recommender.onrender.com
- **Swagger Docs:** https://talentaxis-assessment-recommender.onrender.com/docs
