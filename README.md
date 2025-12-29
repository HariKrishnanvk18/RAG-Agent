# RAG (Retrieval Augmented Generation)
# PDF & DOCUMENT Question Answering System

<img width="1192" height="761" alt="image" src="https://github.com/user-attachments/assets/f52bd058-1b22-48b0-b3d9-61ec1ecc0779" />
<img width="1177" height="796" alt="image" src="https://github.com/user-attachments/assets/fd20d158-003f-43e2-bef3-97f220ffbfdb" />
📄 AI Document Chat (PDF & DOCX Question Answering System)

An AI-powered document chat application that allows users to upload PDF and DOCX files and ask questions strictly based on the uploaded content. The system uses semantic search + confidence-based retrieval to ensure reliable, context-grounded answers and avoid hallucinations.

This project is built using Streamlit, LangChain, and FAISS, following a Retrieval-Augmented Generation (RAG) architecture with intelligent caching and confidence thresholds.

🚀 Features

📤 Upload multiple PDF and DOCX files

🧠 AI-powered question answering using document content only

🔍 Semantic search with FAISS vector embeddings

📊 Confidence-based retrieval (prevents hallucinated answers)

💬 Chat-style conversational interface

♻️ Intelligent answer caching with similarity matching

⚡ Faster responses for repeated or paraphrased questions

❌ Graceful fallback when documents lack relevant information

🛠️ Tech Stack

Frontend / UI: Streamlit

Backend Logic: Python

LLM: OpenAI (gpt-4o-mini)

Embeddings: OpenAI Embeddings

Vector Store: FAISS

Document Parsing:

PDFs → pypdf

DOCX → python-docx

AI Architecture: Retrieval-Augmented Generation (RAG)

🧠 How It Works

User uploads one or more PDF/DOCX documents.

Text is extracted from each document.

Content is split into overlapping chunks for better semantic retrieval.

Each chunk is converted into vector embeddings.

Embeddings are stored in a FAISS vector index.

When a question is asked:

The system checks for cached or similar past questions.

Relevant chunks are retrieved using semantic similarity.

Retrieval confidence is evaluated using distance scores.

The LLM generates an answer only if confidence is sufficient.

High-confidence answers are cached for faster future responses.

🎯 Confidence-Based Answering Logic

High confidence → Answer is generated and cached

Medium confidence → Answer is generated but not cached

Low confidence → The system refuses to answer and asks the user to rephrase

This ensures:

No hallucinated answers

No misleading responses

Clean, document-grounded outputs

🖥️ User Interface Overview

File uploader supporting multiple PDFs/DOCX files

Processing indicator during document indexing

Chat interface displaying:

User queries

AI-generated answers

Input box for document-related questions

Automatic UI refresh for smooth conversation flow

📌 Use Cases

📘 Studying academic notes or textbooks

📄 Analyzing company policies or internal documents

📚 Understanding research papers

🤖 Building AI-powered document assistants

🧠 Knowledge retrieval from private document collections

⚠️ Limitations

Answers are strictly limited to uploaded documents

Requires a valid OpenAI API key

Large documents may increase processing time

Confidence thresholds are embedding-model dependent
