# End-to-End Financial AI for Commodity Risk Analysis

A “Second Brain” for Commodity Risk Analysts using GraphRAG, FinBERT, and Reflection-Based Generation.

---

## Overview

This project presents an end-to-end Financial AI system designed for commodity risk analysis in highly volatile energy markets such as Crude Oil, LNG, and Diesel.

The system autonomously:
- Ingests multi-source commodity news
- Performs commodity-specific sentiment analysis using a fine-tuned FinBERT model
- Retrieves contextual information using Hybrid RAG + GraphRAG
- Generates grounded analytical risk reports using reflection-based generation

The goal is to help analysts shift from manual news monitoring to faster and more structured market intelligence synthesis.

---

## Problem Statement

Energy markets are highly sensitive to:
- Geopolitical conflicts
- Supply disruptions
- Inventory shocks
- Macroeconomic changes

Traditional analyst workflows rely heavily on manual monitoring and delayed reporting, making it difficult to react quickly to fast-moving market events.

This project aims to build a “Second Brain” that:
- Detects important commodity-related signals
- Retrieves relevant contextual information
- Synthesises actionable risk reports automatically

---

## System Architecture

```text
User Query
    ↓
Commodity + Date Extraction
    ↓
News Data Pipeline
(API + Static Datasets)
    ↓
Hybrid RAG Retrieval
(BM25 + Semantic Search + Time Weighting)
    ↓
GraphRAG Knowledge Graph Augmentation
    ↓
LLM Report Generation
(Zero-shot / Few-shot / Citation / Reflection)
    ↓
Fine-tuned FinBERT Sentiment Classification
    ↓
Final Commodity Risk Report
