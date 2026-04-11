import os
import sys
import json
import warnings
from pathlib import Path
from collections import Counter
import torch
import pandas as pd
from dotenv import load_dotenv

# Suppress HuggingFace and PyTorch warnings for clean CLI output
warnings.filterwarnings("ignore")

# ── Setup Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR / "RAG"))
sys.path.insert(0, str(BASE_DIR / "evaluation"))
sys.path.insert(0, str(BASE_DIR / "data"))

from ablation_configs import PIPELINE_CONFIGS
from run_ablation import retrieve_docs
from generation import Generator
from data_loader import load_data

load_dotenv()

# Ask for API key if missing
if not os.environ.get("OPENAI_API_KEY"):
    import getpass
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OPENAI_API_KEY: ")

# ── Load Configuration ──────────────────────────────────────────────────
CONFIG_PATH = BASE_DIR / "evaluation" / "ablation_outputs" / "best_config_selection.json"
DATA_PATH = BASE_DIR / "data" / "final_df.csv"

print("Loading data...")
df = load_data(data_path = DATA_PATH)

documents = []
metadata = []
for _, row in df.iterrows():
    text = (
        f"\n        Title: {row['title']}\n"
        f"        Description: {row['description']}\n"
        f"        Commodity: {row['relevant_commodities']}\n"
        f"        Category: {row['news_category']}\n"
        f"        Risk: {row['risk_category']}\n"
        f"        "
    )
    documents.append(text)
    metadata.append({
        "date"     : row["date"],
        "commodity": row["relevant_commodities"],
        "risk"     : row["risk_category"],
        "severity" : row["risk_severity"],
    })
doc_to_idx = {doc: i for i, doc in enumerate(documents)}

print("Loading best architecture configuration...")
best_key = "E" # Safe default (full system)
if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            scores = json.load(f)
            best_key = max(scores, key=lambda k: scores[k]["combined_score"])
            print(f" > Dynamically loaded best architecture: [{best_key}] (score: {scores[best_key]['combined_score']})")
    except Exception as e:
        print(f" > Warning: Could not read best config ({e}). Defaulting to [{best_key}].")
else:
    print(f" > Warning: {CONFIG_PATH} not found. Did you run the ablation study? Defaulting to [{best_key}].")

config = PIPELINE_CONFIGS[best_key]

# ── Load FinBERT ─────────────────────────────────────────────────────────
FINBERT_DIR = BASE_DIR / "finbert" / "best_finbert"
finbert_available = False
classifier = None
tokenizer = None

print("Loading FinBERT sentiment module...")
if FINBERT_DIR.exists():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch.nn.functional as F
        
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_DIR)
        classifier = AutoModelForSequenceClassification.from_pretrained(FINBERT_DIR)
        classifier.eval()
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier.to(device)
        
        id2label = classifier.config.id2label
        if not id2label or 0 not in id2label:
            id2label = {0: "Bullish", 1: "Bearish", 2: "Neutral"}
            
        finbert_available = True
        print(f" > Successfully loaded fine-tuned FinBERT on {device}.")
    except Exception as e:
        print(f" > Fallback Warning: Failed to load FinBERT ({e}). Falling back to LLM sentiment generation.")
else:
    print(f" > Fallback Warning: Custom FinBERT model not found at {FINBERT_DIR}. "
          "Falling back to standard LLM sentiment deduction.")

def predict_sentiment(texts: list[str]) -> str:
    """Predicts consensus sentiment using FinBERT"""
    if not finbert_available or not texts:
        return None
        
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(classifier.device)

    with torch.no_grad():
        outputs = classifier(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
    
    labels = [id2label[int(p)].capitalize() for p in preds]
    
    # Simple majority consensus
    counts = Counter(labels)
    consensus_label, _ = counts.most_common(1)[0]
    
    # Build a strictly explicit string for the LLM
    details = ", ".join([f"{count} {label}" for label, count in counts.items()])
    return f"{consensus_label} (Consensus breakdown: {details})"

# ── Interactive CLI Loop ─────────────────────────────────────────────────
generator = Generator(strategy="citation", model="gpt-4o-mini")

print("\n" + "="*70)
print("  RAG + FinBERT Market Risk Analysis System initialized.")
print("  Type your query below. Type 'exit' or 'quit' to close.")
print("="*70 + "\n")

while True:
    try:
        query = input("\nQuery: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting...")
            break
            
        print(" > Retrieving documents...")
        docs = retrieve_docs(query, documents, metadata, doc_to_idx, config)
        
        if not docs:
            print(" > No relevant documents found. Please broaden your query criteria (e.g. adjust dates).")
            continue
            
        # Optional GraphRAG Augmentation
        subgraph = []
        if config.use_graph_rag:
            print(" > Building GraphRAG context...")
            from graph_rag import GraphRAG
            graph = GraphRAG()
            docs_for_graph = []
            for d in docs:
                idx = doc_to_idx.get(d["text"])
                if idx is None:
                    continue
                entities = [
                    metadata[idx]["commodity"],
                    df.iloc[idx]["news_category"],
                    df.iloc[idx]["risk_category"],
                ]
                docs_for_graph.append({"text": d["text"], "entities": entities})
            graph.build_graph(docs_for_graph)
            
            # Simple query extraction
            from metadata_filtering import QueryProcessor
            qp = QueryProcessor()
            parsed = qp.parse_query(query)
            q_entities = []
            if parsed["commodity"]:
                q_entities.append(parsed["commodity"])
            q_entities.extend(query.lower().split())
            subgraph = graph.retrieve_subgraph(q_entities)
        
        finbert_consensus = None
        if finbert_available:
            print(" > Inferring market sentiment via FinBERT...")
            texts = [d["text"] for d in docs]
            finbert_consensus = predict_sentiment(texts)
            if finbert_consensus:
                print(f"   [FinBERT Output]: {finbert_consensus}")
            
        print(" > Generating report...")
        report = generator.generate(
            query=query, 
            reranked_docs=docs, 
            graph_context=subgraph,
            finbert_consensus=finbert_consensus
        )
        
        print("\n" + "="*70)
        print(report)
        print("="*70)
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\n[Error] {e}")
