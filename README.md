🧪 AI Lab Compliance Copilot
Intelligent GMP Deviation & SOP Assistant with RAG + LLM Pipeline
📘 Overview
 
AI Lab Compliance Copilot is an intelligent pharmaceutical QA system that automates SOP question-answering, deviation reporting, and trend analysis using a Retrieval-Augmented Generation (RAG) framework.
It connects GMP compliance data (SOP PDFs + deviation logs) to a Groq LLM for context-aware report generation.
 
🏗️ System Architecture
 
🔹 Core Workflow
 
Streamlit Frontend → User inputs queries / incident descriptions.
 
FastAPI Backend (BE/) → Handles requests and interfaces with Redis + Groq LLM.
 
PDF Processor → Extracts and embeds text chunks from SOP documents.
 
SentenceTransformer Embeddings → Converts text into vectors for semantic search.
 
Redis Vector Store (RAG Memory) → Retrieves relevant SOP contexts on query.
 
Groq LLM (LLaMA-3.3) → Generates answers and deviation analysis using retrieved contexts.
 
ReportLab PDF Generator → Creates PDF summaries and trend reports.
 
🧠 RAG Pipeline
graph TD
A[SOP PDF Upload] --> B[Text Extraction & Chunking]
B --> C[SentenceTransformer Embeddings]
C --> D[Redis Vector Index]
D --> E[User Query]
E --> F[Vector Search & Top-K Context Retrieval]
F --> G[Groq LLM (GMP Reasoning)]
G --> H[Deviation Report / SOP Answer + PDF Generation]
 
⚙️ Features
Feature	Description
💬 SOP Chat (RAG)	Ask SOP-based questions; system retrieves relevant SOP sections and generates answers via LLM.
🚨 Deviation Reporting	Report incidents with severity and category. Generates LLM-based analysis and PDF report.
📊 Trend Analysis	Aggregates deviation data and identifies patterns for training recommendations.
🎓 Retraining Suggestions	Automatically recommends SOP retraining areas and urgency levels.
📄 PDF Export	Generates auditable reports for QA review and documentation.
🧩 Tech Stack
Layer	Technology
Frontend	Streamlit
Backend	FastAPI
Vector DB	Redis Stack (Search + JSON Modules)
Embeddings	SentenceTransformer (multi-qa-mpnet-base-dot-v1)
LLM Engine	Groq LLM (LLaMA-3.3-70B Versatile)
Report Generation	ReportLab
Containerization (Optional)	Docker + Redis Stack Image
📂 Folder Structure
project_root/
│
├── BE/                         # FastAPI backend
│   ├── pharma_devbot_full.py    # Main backend file with RAG logic
│   ├── pdf_generator.py         # PDF creation for deviation reports
│   ├── requirements.txt
│   └── sop_docs/                # Uploaded SOP PDF files
│
├── frontend/                    # Streamlit frontend dashboard
│   └── app.py                   # UI for SOP chat + Deviation Reports
│
├── data/                        # Processed data / embeddings
├── uploads/                     # User-uploaded PDFs
├── bot.py                       # Chat handling module
├── llm.py                       # LLM interface for Groq API
├── query.py                     # SOP search and context retrieval
├── pdf_processing.py             # PDF text extraction and embedding
└── README.md
 
🚀 Quick Start
1️⃣ Run Redis Stack
docker run -d -p 6379:6379 redis/redis-stack:latest
 
2️⃣ Backend Setup
cd BE
pip install -r requirements.txt
uvicorn pharma_devbot_full:app --reload --port 8000
 
3️⃣ Frontend Setup
cd ../frontend
streamlit run app.py
 
4️⃣ Usage
 
Upload SOP PDFs in the sidebar.
 
Ask SOP questions in chat tab.
 
Create Deviation Reports via Incident tab.
 
View Trends and Training under Analytics tabs.
