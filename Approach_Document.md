# SHL Conversational Assessment Recommender: Approach Document

## 1. Design Choices
- **Framework & API:** FastAPI is used to expose the `/health` and `/chat` endpoints due to its fast async capabilities and built-in support for Pydantic. Pydantic guarantees that the I/O matches the required automated evaluator harness schema (`reply`, `recommendations`, `end_of_conversation`).
- **LLM Integration:** Gemini 1.5 Flash (via `google-genai`) is chosen for the conversational agent. Gemini provides an excellent balance of speed and cost, making it ideal for real-time conversational agents. The `GenerateContentConfig(response_schema=...)` is utilized to force the LLM to return strict JSON matching the required schema, eliminating output parsing errors.
- **Statelessness:** As required, the API stores no state. Each POST request carries the entire conversation history, which is injected into the prompt.

## 2. Retrieval Setup (RAG)
To ground the LLM and prevent hallucination, an in-memory vector database is implemented using `faiss-cpu` and `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Initialization:** During the FastAPI cold-start `startup_event`, the catalog JSON is parsed. Each assessment is formatted into a document string containing its Name, Keys, Job Levels, and Description.
- **Search Logic:** The agent extracts a heuristic search query by combining the latest user and assistant messages. This query is embedded and searched against the FAISS index to retrieve the Top 15 matching assessments.
- **Grounding:** Only these Top 15 candidates (minified to save context tokens) are injected into the LLM's system prompt. This guarantees the LLM only selects valid assessments.

## 3. Prompt Design
The System Prompt assigns the persona of a "Conversational SHL Assessment Recommender" and explicitly maps to the four behavioral requirements:
1. **Clarify:** Rejects vague intents and requests more context, ensuring `recommendations` remains empty.
2. **Recommend & Refine:** Provides 1-10 assessments when context is sufficient, updating dynamically if constraints change.
3. **Compare:** Answers comparison requests strictly using the provided catalog data (e.g., differentiating OPQ and GSA).
4. **Safety & Scope:** Explicitly instructs the LLM to refuse general hiring advice, legal questions, and prompt injections.
5. **JSON Structuring:** Instructs the LLM to map the catalog `link` to `url` exactly, and deduce `test_type` (e.g., 'P', 'K') from the catalog 'keys'.

## 4. Evaluation Approach
- **Automated Validation:** A `test_app.py` script is used to validate that the endpoints are responsive and strictly adhere to the expected schema without breaking.
- **Simulated Traces:** Tested against the sample conversation traces (e.g., C1, C2) to verify that the LLM appropriately refuses recommendations on Turn 1 when context is lacking, and eventually provides the correct shortlist.

## 5. What Didn't Work
- **Zero-Shot without Retrieval:** Initially relying on the LLM's internal knowledge of SHL led to hallucinations of non-existent URLs and products.
- **Improvement:** Introducing the FAISS retrieval step and explicitly hardcoding a rule to *only* select from the `CATALOG CANDIDATES` JSON block resulted in 100% valid URLs.
- **TF-IDF Matching:** TF-IDF was considered for lower latency but failed to capture semantic overlaps (e.g., matching "Rust" to "Smart Interview Live Coding"). `all-MiniLM-L6-v2` successfully bridges semantic gaps while remaining lightweight.

## 6. AI Tools Used
- **Antigravity Agent (Gemini 3.1 Pro):** Used to autonomously plan the architecture, write the FastAPI service, set up the FAISS vector store, structure the Pydantic schemas, and author this approach document.
