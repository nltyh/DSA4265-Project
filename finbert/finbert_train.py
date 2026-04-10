# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:28:46 2026

@author: grace
"""
#%%
#load packages
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed
)
#%%
df = pd.read_csv("data/CrudeOilNewsData_finbert_train_ready.csv")

market_label2id = {
    "bullish": 0,
    "bearish": 1,
    "neutral": 2
}

df["market_impact_label"] = df["market_impact_label"].astype(str).str.strip().str.lower()
df["market_impact_label_id"] = df["market_impact_label"].map(market_label2id)
#%%
#config
MODEL_NAME = "ProsusAI/finbert"
TEXT_COL = "text"
LABEL_COL = "market_impact_label_id"
OUTPUT_DIR = os.getcwd()

MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUM_LABELS = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

# Label mapping
id2label = {
    0: "bullish",
    1: "bearish",
    2: "neutral"
}
label2id = {v: k for k, v in id2label.items()}
#%%
#set seed
def seed_everything(seed=19):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

seed_everything(RANDOM_STATE)
#%%
#tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples[TEXT_COL],
        truncation=True,
        max_length=MAX_LENGTH
    )
#%%
#handle data
# Keep only needed columns
df = df[[TEXT_COL, LABEL_COL]].copy()

# Clean rows
df = df.dropna(subset=[TEXT_COL, LABEL_COL])
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
df = df[df[TEXT_COL] != ""]

# Make sure labels are integers
df[LABEL_COL] = df[LABEL_COL].astype(int)

# train / validation split
train_df, val_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df[LABEL_COL]
)

train_df = train_df.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True)
val_df = val_df.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True)


class FinBERTDataset(TorchDataset):
    def __init__(self, dataframe, tokenizer, text_col="text", label_col="labels", max_length=256):
        self.texts = dataframe[text_col].astype(str).tolist()
        self.labels = dataframe[label_col].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = FinBERTDataset(
    train_df,
    tokenizer=tokenizer,
    text_col="text",
    label_col="labels",
    max_length=MAX_LENGTH
)

val_dataset = FinBERTDataset(
    val_df,
    tokenizer=tokenizer,
    text_col="text",
    label_col="labels",
    max_length=MAX_LENGTH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

model.classifier = nn.Linear(model.config.hidden_size, NUM_LABELS)
model.classifier.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
model.classifier.bias.data.zero_()


# METRICS

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "weighted_f1": f1
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

#evaluate
eval_results = trainer.evaluate()

for k, v in eval_results.items():
    print(f"{k}: {v}")
    
#%%
#checks
def predict_texts(texts):
    model.eval()
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()
        probs = probs.cpu().numpy()

    results = []
    for text, pred, prob in zip(texts, preds, probs):
        results.append({
            "text": text,
            "predicted_label_id": int(pred),
            "predicted_label": id2label[int(pred)],
            "confidence": float(np.max(prob))
        })
    return results


sample_texts = [
    "OPEC+ signals production cuts, raising concerns about tighter crude supply.",
    "US crude inventories unexpectedly rose this week, pressuring oil prices.",
    "Energy markets were mostly unchanged as traders awaited new macro data."
]

print("\nSample predictions:")
for row in predict_texts(sample_texts):
    print(row)

model.save_pretrained("best_finbert")
tokenizer.save_pretrained("best_finbert")