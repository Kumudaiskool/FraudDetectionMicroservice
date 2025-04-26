# FraudDetectionMicroservice

# 💡 GenAI-Powered Fraud Detection System

This repository contains a proof-of-concept fraud detection framework that leverages GenAI techniques, including OpenAI's GPT models, FinBERT, and Retrieval-Augmented Generation (RAG) pipelines for advanced anomaly detection and transaction analysis.

---

## 🚀 Project Overview

The financial world is evolving — so are fraud patterns. Traditional fraud detection rules often miss novel or subtle fraud strategies. This project demonstrates how Large Language Models (LLMs) combined with domain-specific NLP models can enhance fraud detection by reasoning about suspicious activities rather than relying on static patterns.

---

## 🔍 Features

- GPT-4 based transaction reasoning.
- FinBERT sentiment analysis for transaction narratives.
- Retrieval-Augmented Generation (RAG) pipeline using ChromaDB.
- Multi-agent collaboration simulation for fraud scenario analysis.
- API-ready with FastAPI for easy integration.
- Demonstrated on Kaggle's synthetic fraud dataset.

---

## 📦 Tech Stack

- Python 3.10
- OpenAI GPT-4 API
- HuggingFace Transformers (FinBERT)
- ChromaDB for vector search
- FastAPI for service deployment
- LangChain for agent orchestration
- Docker (for containerized deployments)

---

## 💡 How It Works

1. **Data Ingestion:**  
Transaction datasets are processed through a pre-cleaning pipeline.

2. **Sentiment Pre-Screening:**  
FinBERT is used to assess transaction text sentiment, identifying potential negative or ambiguous cues.

3. **LLM Fraud Reasoning:**  
Transactions flagged for review are passed to GPT-4, which applies contextual reasoning to evaluate fraud likelihood.

4. **RAG Support:**  
The system retrieves similar historical transactions from a vector database to aid LLM decision-making.

---

## 🧪 Sample Dataset

Using [Kaggle's Synthetic Financial Dataset](https://www.kaggle.com/ealaxi/paysim1) for testing.

---

## 📂 Run Locally

```bash
git clone https://github.com/yourusername/genai-fraud-detection.git
cd genai-fraud-detection
pip install -r requirements.txt
uvicorn main:app --reload
