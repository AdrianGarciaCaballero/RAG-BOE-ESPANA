import json
import logging
import pandas as pd
from retrieval_engine import RetrievalEngine
import os
import sys

# Setup logging
logging.basicConfig(level=logging.ERROR) # Only errors to keep output clean
logger = logging.getLogger(__name__)

def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)

def evaluate_config(engine, dataset, top_k=5, config_name="Config A"):
    print(f"\n🚀 Evaluando: {config_name} (Top-K={top_k})...")
    
    hits = 0
    mrr_sum = 0
    total = len(dataset)
    
    results_detail = []

    for item in dataset:
        question = item["question"]
        expected_doc = item["reference_doc"]
        
        # Retrieval (Hybrid + Rerank)
        # We simulate different configs by changing top_k in the final step or retrieval
        try:
             # Get more candidates first to allow reranker to work
            hybrid_candidates = engine.hybrid_search(question, top_k_fusion=20) 
            final_results = engine.rerank(question, hybrid_candidates, top_k=top_k)
            
            # Check correctness
            found = False
            rank = 0
            
            retrieved_docs_names = []
            
            for i, res in enumerate(final_results):
                doc_name = res['metadata'].get('source', '')
                retrieved_docs_names.append(doc_name)
                
                # Check match (exact filename)
                if expected_doc == doc_name:
                    found = True
                    rank = i + 1
                    break
            
            if found:
                hits += 1
                mrr_sum += 1.0 / rank
                
            results_detail.append({
                "Question": question,
                "Expected": expected_doc,
                "Found": found,
                "Rank": rank
            })
            
        except Exception as e:
            print(f"Error processing '{question}': {e}")
            
    hit_rate = hits / total
    mrr = mrr_sum / total
    
    return {
        "Config": config_name,
        "Hit Rate": hit_rate,
        "MRR": mrr
    }

def main():
    print("📋 Iniciando Evaluación de Retrieval...")
    
    # Initialize Engine
    BASE_DIR = os.getcwd()
    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
    COLLECTION_NAME = "rag_multimodal"
    
    engine = RetrievalEngine(CHROMA_PATH, COLLECTION_NAME)
    
    dataset = load_dataset()
    
    # Run Comparison (Requirement: Comparative Table)
    # Config 1: Strict (Top-3)
    res1 = evaluate_config(engine, dataset, top_k=3, config_name="Top-3 (Strict)")
    
    # Config 2: Broad (Top-10)
    res2 = evaluate_config(engine, dataset, top_k=10, config_name="Top-10 (Broad)")
    
    # Create DataFrame
    df = pd.DataFrame([res1, res2])
    
    print("\n\n📊 TABLA COMPARATIVA DE RESULTADOS (Retrieval):")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    df.to_csv("retrieval_metrics.csv", index=False)
    print("💾 Resultados guardados en 'retrieval_metrics.csv'")

if __name__ == "__main__":
    main()
