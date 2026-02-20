import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
CSV_FILE = r"C:\Users\USER\Desktop\llm project\AAPL_Stock_Price_Dataset.csv"   # <-- Update path
DB_LOCATION = "./chroma_amzn_stock_db"
COLLECTION_NAME = "amzn_stock_history"
BATCH_SIZE = 1000

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

# Ensure Date column is string (important for metadata consistency)
df["Date"] = df["Date"].astype(str)

# =========================
# EMBEDDING MODEL
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# VECTOR STORE
# =========================
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

existing_count = vector_store._collection.count()
print("Existing documents in DB:", existing_count)

# =========================
# PREPARE DOCUMENTS
# =========================
documents = []
ids = []

for i, row in df.iterrows():

    # Handle optional columns safely
    adj_close = row.get("Adj Close", "N/A")
    market_cap = row.get("Market Cap", "N/A")
    corporate_action = row.get("Split/Dividend", "None")

    content = (
        f"On {row['Date']}, Amazon (AMZN) stock opened at {row['Open']} USD, "
        f"reached a high of {row['High']} USD, "
        f"dropped to a low of {row['Low']} USD, "
        f"and closed at {row['Close']} USD. "
        f"The adjusted close price was {adj_close} USD. "
        f"Trading volume was {row['Volume']}. "
        f"Estimated market capitalization was {market_cap}. "
        f"Corporate action details: {corporate_action}."
    )

    doc = Document(
        page_content=content,
        metadata={
            "ticker": "AMZN",
            "date": row["Date"],
            "year": row["Date"][:4],
        },
        id=str(i)
    )

    documents.append(doc)
    ids.append(str(i))

# =========================
# INGEST (ONLY IF EMPTY)
# =========================
if existing_count == 0:
    print("Ingesting Amazon stock documents into Chroma...")

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

        print(f"Inserted documents {i} to {i + len(batch_docs)}")

    print("Final document count:", vector_store._collection.count())
else:
    print("Using existing embeddings. No re-ingestion needed.")

# =========================
# RETRIEVER
# =========================
retriever = vector_store.as_retriever(
    search_kwargs={"k": 15}
)
