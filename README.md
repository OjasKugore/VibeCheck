# review-sentiment-app
# VibeCheck: Real-Time Sentiment Intelligence System

VibeCheck is an advanced AI-powered application designed to solve "Analysis Paralysis" in consumer decision-making. By synthesizing real-time web reviews into a concise, actionable sentiment report, VibeCheck bridges the gap between fragmented consumer feedback and objective decision-making.

## Key Features

* **Real-Time RAG (Retrieval-Augmented Generation):** Unlike static AI models, VibeCheck fetches the latest review data from across the web using the Serper API, ensuring insights are always current.
* **Aspect-Based Sentiment Analysis (ABSA):** Automatically extracts specific Pros and Cons, distilling noisy text into structured, high-value insights.
* **Signal Strength Quantification:** Provides transparency by reporting the amount of data analyzed, allowing users to gauge the reliability of the verdict.
* **Advanced Sentiment Engine:** Powered by Google Gemini 3 Flash, the system performs semantic reasoning to detect sarcasm, context, and nuance, moving beyond simple keyword counting.
* **Zero-Latency Pipeline:** Built on a browserless, REST-based API architecture for rapid performance.

## Technology Stack

* **Frontend & Backend:** [Streamlit](https://streamlit.io/) (Python)
* **Intelligence Engine:** [Google Gemini 3 Flash](https://ai.google.dev/)
* **Data Retrieval:** [Serper.dev API](https://serper.dev/)
* **Visualization:** [Plotly](https://plotly.com/)
* **Deployment:** Streamlit Cloud

## Architecture Overview

1.  **Retrieval:** User input triggers the Serper API to scrape top-tier review snippets from diverse sources (Reddit, Amazon, Tech Blogs).
2.  **Augmentation:** Retrieved snippets are aggregated into a structured prompt, providing the AI with the necessary context ("grounding").
3.  **Generation:** The Gemini 3 Flash model parses this context to output a structured JSON verdict (Sentiment Score, Vibe Summary, Pros, and Cons).
4.  **Presentation:** Streamlit visualizes the sentiment using dynamic gauge charts and responsive UI components.

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/vibecheck.git
    cd vibecheck
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment Variables:**
    Create a `.env` file and add your API keys:
    ```text
    GEMINI_API_KEY=your_gemini_key_here
    SERPER_API_KEY=your_serper_key_here
    ```
4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## Risk & Mitigation
* **Hallucinations:** Mitigated by strict RAG grounding and JSON-only structural constraints.
* **API Limits:** Handled via exponential backoff retry logic and intelligent model fallback.
* **Data Bias:** Addressed by pulling from multi-platform sources to achieve a consensus-driven sentiment.

##  Academic Context
Developed as an engineering project exploring the frontiers of **Natural Language Understanding (NLU)** and **Dynamic Information Retrieval**. VibeCheck demonstrates the efficiency of modern LLMs in high-speed, real-world data processing tasks.
