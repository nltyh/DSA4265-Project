import pandas as pd
from pathlib import Path
import json

from hybrid_retriever import HybridRetriever
from reranker import Reranker
from time_weighter import TimeWeighter
from graph_rag import GraphRAG
from metadata_filtering import QueryProcessor, MetadataFilter
from generation import Generator, build_context
from evaluate import evaluate_reports, print_summary


# ---- Load Data ----
def load_data(file_path):
    df = pd.read_csv(file_path)

    documents = []
    metadata = []

    for _, row in df.iterrows():
        text = f"""
        Title: {row['title']}
        Description: {row['description']}
        Commodity: {row['relevant_commodities']}
        Category: {row['news_category']}
        Risk: {row['risk_category']}
        """

        documents.append(text)

        metadata.append({
            "date": row["date"],
            "commodity": row["relevant_commodities"],
            "risk": row["risk_category"],
            "severity": row["risk_severity"]
        })

    # Create fast lookup
    doc_to_idx = {doc: i for i, doc in enumerate(documents)}

    return df, documents, metadata, doc_to_idx


# ---- Pipeline ----
def run_pipeline(query, documents, metadata, df, doc_to_idx, generation_strategy="citation"):
    print("\n🔎 QUERY:", query)

    # ---- 0. Query Parsing ----
    qp = QueryProcessor()
    parsed = qp.parse_query(query)

    print("Parsed Query:", parsed)

    # ---- 1. Metadata Filtering ----
    filter = MetadataFilter(
        target_date=parsed["date"],
        window_days=parsed["window_days"],
        commodity=parsed["commodity"]
    )

    filtered_docs, filtered_meta = filter.apply(documents, metadata)

    print(f"Filtered Docs: {len(filtered_docs)} / {len(documents)}")

    # ---- 2. Hybrid Retrieval ----
    retriever = HybridRetriever(filtered_docs)
    hybrid_results = retriever.hybrid_search(query, top_k=10, alpha=0.7)

    # ---- 3. Attach Metadata ----
    results_with_meta = []
    for doc, score in hybrid_results:
        idx = doc_to_idx[doc]

        results_with_meta.append({
            "text": doc,
            "score": score,
            "date": metadata[idx]["date"],
            "commodity": metadata[idx]["commodity"]
        })

    # ---- 4. Time Weighting ----
    time_weighter = TimeWeighter(decay_rate=0.03)
    time_weighted = time_weighter.apply(results_with_meta)

    # ---- 5. Reranking ----
    reranker = Reranker()

    texts = [d["text"] for d in time_weighted]
    pairs = [(query, t) for t in texts]

    scores = reranker.model.predict(pairs)

    reranked = []
    for i, score in enumerate(scores):
        reranked.append({
            "text": texts[i],
            "score": score,
            "date": time_weighted[i]["date"],
            "commodity": time_weighted[i]["commodity"]
        })

    # Sort
    reranked.sort(key=lambda x: x["score"], reverse=True)

    # ---- 6. GraphRAG ----
    graph = GraphRAG()

    docs_for_graph = []
    for d in time_weighted:
        idx = doc_to_idx[d["text"]]

        entities = [
            metadata[idx]["commodity"],
            df.iloc[idx]["news_category"],
            df.iloc[idx]["risk_category"]
        ]

        docs_for_graph.append({
            "text": d["text"],
            "entities": entities
        })

    graph.build_graph(docs_for_graph)

    #  Query entities (use parsed commodity if available)
    query_entities = []
    if parsed["commodity"]:
        query_entities.append(parsed["commodity"])
    query_entities.extend(query.lower().split())

    subgraph = graph.retrieve_subgraph(query_entities)

    # ---- 7. Generation ----
    print(f"\n✍️  Generating report with strategy: {generation_strategy}")
    gen = Generator(strategy=generation_strategy)
    report = gen.generate(query, reranked, graph_context=subgraph)
    generation_docs = reranked[: gen.top_k_docs]

    # ---- OUTPUT ----
    print("\n📊 Top Reranked Results:")
    for item in reranked[:5]:
        print(f"\nScore: {item['score']:.4f}")
        print(f"Date: {item['date']} | Commodity: {item['commodity']}")
        print(item['text'][:200], "...")

    print("\n🌐 Graph Context:")
    print(subgraph)

    print("\n📝 Generated Report:")
    print(report)

    return {
        "report": report,
        "generation_docs": generation_docs,
    }


# ---- MAIN ----
if __name__ == "__main__":
    file_path = "data/merged_df_v2.csv"

    df, documents, metadata, doc_to_idx = load_data(file_path)

    queries = [
        "After Russia invaded Ukraine, provide crude oil market snapshot" # CHANGE THIS TO TEST OTHER QUERIES
    ]

    STRATEGY = "citation"
    EVALUATE_REPORT = True

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for i, q in enumerate(queries, start=1):
        result = run_pipeline(q, documents, metadata, df, doc_to_idx,
                              generation_strategy=STRATEGY)
        output_path = output_dir / f"query_{i}_{STRATEGY}_report.txt"
        output_path.write_text(result["report"], encoding="utf-8")
        print(f"\n💾 Saved report to {output_path}")

        if EVALUATE_REPORT:
            context = build_context(result["generation_docs"])
            evaluation = evaluate_reports(
                query=q,
                reports={STRATEGY: result["report"]},
                context=context,
                reference=None,
                run_groundedness=True,
                run_llm_judge=True,
            )
            print_summary(evaluation)

            evaluation_path = output_dir / f"query_{i}_{STRATEGY}_evaluation.json"
            evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
            print(f"\n💾 Saved evaluation to {evaluation_path}")
