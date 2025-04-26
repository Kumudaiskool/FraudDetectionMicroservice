from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize a FastAPI instance for the web service
app = FastAPI()

# Set your OpenAI API key to authenticate OpenAI requests
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load the FinBERT tokenizer and model, pre-trained for financial sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Define the schema for incoming transaction data using Pydantic
class Transaction(BaseModel):
    id: str               # Unique identifier for the transaction
    description: str      # Description text of the transaction
    amount: float         # Monetary amount involved in the transaction
    metadata: dict        # Additional transaction-related metadata

# Function to detect sentiment in transaction descriptions using FinBERT
def detect_sentiment(transaction_text: str) -> str:
    print(f"[INFO] Running FinBERT sentiment detection for text: {transaction_text}")

    # Tokenize the input text for the FinBERT model
    inputs = tokenizer(transaction_text, return_tensors="pt", truncation=True, max_length=512)
    print(f"[DEBUG] Tokenized inputs: {inputs}")

    # Run the model forward pass to get raw logits
    outputs = model(**inputs)
    print(f"[DEBUG] Model raw outputs: {outputs}")

    # Apply softmax to transform logits into probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(f"[DEBUG] Softmax probabilities: {probs}")

    # Identify the class with the highest probability
    sentiment = torch.argmax(probs).item()
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    detected_sentiment = sentiment_map[sentiment]
    print(f"[INFO] Detected sentiment: {detected_sentiment}")

    return detected_sentiment

# Asynchronous function to request fraud reasoning from OpenAI's GPT-4 model
async def gpt4_fraud_reasoning(transaction: Transaction) -> str:
    print(f"[INFO] Starting GPT-4 fraud reasoning for transaction ID: {transaction.id}")

    # Construct a clear prompt for GPT-4 to evaluate the transaction's fraud likelihood
    prompt = f"""You are an AI fraud analyst. Analyze the following transaction and assess if it could be fraudulent:\n\nTransaction ID: {transaction.id}\nDescription: {transaction.description}\nAmount: ${transaction.amount}\nMetadata: {transaction.metadata}\n\nProvide a probability estimate and reasoning."""
    print(f"[DEBUG] GPT-4 prompt: {prompt}")

    # Query GPT-4 via OpenAI's API
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Lower temperature ensures more deterministic output
    )

    # Extract the content from the response
    result_content = response.choices[0].message['content']
    print(f"[INFO] GPT-4 response: {result_content}")

    return result_content

# API endpoint to accept a list of transactions and return fraud detection analysis
@app.post("/detect-fraud")
async def detect_fraud(transactions: List[Transaction]):
    print(f"[INFO] Received {len(transactions)} transactions for fraud analysis.")
    results = []

    # Iterate over each transaction for analysis
    for txn in transactions:
        print(f"[INFO] Processing transaction ID: {txn.id}")

        # Step 1: Run sentiment analysis on the transaction description
        sentiment = detect_sentiment(txn.description)

        # Step 2: Use GPT-4 to reason about fraud potential
        gpt4_result = await gpt4_fraud_reasoning(txn)

        # Combine and store both sentiment and GPT-4 results
        results.append({
            "transaction_id": txn.id,
            "finbert_sentiment": sentiment,
            "gpt4_analysis": gpt4_result
        })

        print(f"[INFO] Completed analysis for transaction ID: {txn.id}")

    print(f"[INFO] Fraud analysis completed for all transactions.")
    return {"fraud_analysis": results}

# Note for deployment:
# Run this FastAPI service locally for testing using:
# uvicorn filename:app --reload
# Replace 'filename' with the name of this Python script.
