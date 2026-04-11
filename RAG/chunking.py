import pandas as pd
from pathlib import Path
from transformers import GPT2TokenizerFast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Load Data ----
file_path = "data/final_df.csv"
df = pd.read_csv(file_path)

# ---- Initialize GPT tokenizer ----
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# ---- Parameters ----
MAX_TOKENS_PER_CHUNK = 1024  # safe chunk size for GPT-4o
descriptions = df['description'].tolist()

# ---- Stats containers ----
token_counts = []
needs_chunking_flags = []

# ---- Analyze each description ----
for i, desc in enumerate(descriptions):

    # Count tokens
    tokens = tokenizer.encode(desc)
    token_count = len(tokens)
    token_counts.append(token_count)

    # Check if chunking is needed
    needs_chunking = token_count > MAX_TOKENS_PER_CHUNK
    needs_chunking_flags.append(needs_chunking)

    print(f"Doc {i}: tokens={token_count}, needs_chunking={needs_chunking}")

# ---- Summary ----
total_docs = len(descriptions)
chunk_needed_count = sum(needs_chunking_flags)
print("\n📊 Summary:")
print(f"Total documents: {total_docs}")
print(f"Documents needing chunking (> {MAX_TOKENS_PER_CHUNK} tokens): {chunk_needed_count} ({chunk_needed_count/total_docs*100:.2f}%)")

# ---- Plot distributions ----
plt.figure(figsize=(12,5))
plt.hist(token_counts, bins=20, color='salmon', edgecolor='black')
plt.title("Token count distribution")
plt.xlabel("Tokens per document")
plt.ylabel("Number of documents")

plt.tight_layout()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
plot_path = OUTPUT_DIR / "token_distribution.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n📊 Plot saved → {plot_path}")