import os
import logging
import chromadb
import torch
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder, SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import string
from functools import lru_cache

# Configuracion
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Logger
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Limpieza básica para BM25: minusculas y eliminar puntuacion."""
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

class RetrievalEngine:
    _instance = None

    def __new__(cls, chroma_path, collection_name):
        if cls._instance is None:
            cls._instance = super(RetrievalEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, chroma_path, collection_name):
        if self.initialized:
            return
            
        logger.info("⚙️ Iniciando Retrieval Engine (Hybrid + Rerank)...")
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # 1. Cargar Embedding (Bi-Encoder)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME,
            device=device
        )
        
        # 2. Conectar Chroma
        self.client = chromadb.PersistentClient(path=self.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.emb_fn
        )
        
        # 3. Inicializar BM25 (Lazy load)
        self.bm25 = None
        self.bm25_corpus = [] # [(id, text, metadata), ...]
        self._build_bm25_index()
        
        # 4. Inicializar Cross-Encoder (Reranker)
        logger.info(f"⏳ Cargando Reranker: {RERANKER_MODEL_NAME} (Puede tardar la primera vez)...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
            logger.info("✅ Reranker cargado.")
        except Exception as e:
            logger.error(f"❌ Error cargando Reranker: {e}")
            self.reranker = None
            
        self.initialized = True

    def _build_bm25_index(self):
        """Reconstruye el índice BM25 desde ChromaDB (Memoria)."""
        logger.info("🏗️ Construyendo índice BM25...")
        try:
            # Traer TODOs los documentos (Limitación: Si son millones, esto explota. Para miles va bien.)
            data = self.collection.get()
            docs = data['documents']
            ids = data['ids']
            metas = data['metadatas']
            
            if not docs:
                logger.warning("⚠️ ChromaDB vacía. BM25 no indexará nada.")
                return

            self.bm25_corpus = []
            tokenized_corpus = []
            
            for doc_id, text, meta in zip(ids, docs, metas):
                cleaned = clean_text(text)
                tokenized_corpus.append(cleaned.split())
                self.bm25_corpus.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": meta
                })
                
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ BM25 Indexado: {len(self.bm25_corpus)} documentos.")
            
        except Exception as e:
            logger.error(f"❌ Error construyendo BM25: {e}")

    def refresh_bm25(self):
        """Llamar despues de ingestas nuevas."""
        self._build_bm25_index()

    def search_bm25(self, query: str, top_k=20):
        if not self.bm25:
            return []
            
        tokenized_query = clean_text(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Obtener indices con mejores scores
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_indices:
            score = scores[idx]
            if score > 0: # Solo relevantes
                doc_info = self.bm25_corpus[idx]
                results.append({
                    "id": doc_info["id"],
                    "document": doc_info["text"],
                    "metadata": doc_info["metadata"],
                    "score": score
                })
        return results

    def search_vector(self, query: str, top_k=20):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        
        formatted = []
        if results['ids']:
            ids = results['ids'][0]
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            # Chroma distances are dissimilarity usually, but langgraph uses cosine similarity?
            # actually collection.query returns distances. Smaller is better if L2.
            # Using distances as inverse score proxy locally or just rank position.
            
            for i in range(len(ids)):
                formatted.append({
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "score": 0.0 # Vector rank es implícito por orden
                })
        return formatted

    def reciprocal_rank_fusion(self, results_lists, k=60):
        """Combina listas de resultados usando RRF."""
        rrf_map = {}
        
        for result_list in results_lists:
            for rank, item in enumerate(result_list):
                doc_id = item['id']
                if doc_id not in rrf_map:
                    rrf_map[doc_id] = {"score": 0, "item": item}
                
                # Formula: 1 / (k + rank)
                rrf_map[doc_id]["score"] += 1 / (k + rank + 1)
        
        # Convertir a lista ordenada
        sorted_results = sorted(rrf_map.values(), key=lambda x: x['score'], reverse=True)
        return [x['item'] for x in sorted_results]

    def hybrid_search(self, query: str, top_k_fusion=10):
        logger.info(f"🔎 Hybrid Search: '{query}'")
        
        # 1. Parallel Search (Simulated)
        res_bm25 = self.search_bm25(query, top_k=top_k_fusion*2)
        res_vec = self.search_vector(query, top_k=top_k_fusion*2)
        
        # 2. Fusion
        fused = self.reciprocal_rank_fusion([res_bm25, res_vec])
        return fused[:top_k_fusion]

    def rerank(self, query: str, candidates: list, top_k=5):
        if not self.reranker or not candidates:
            return candidates[:top_k]
            
        logger.info(f"⚖️ Reranking {len(candidates)} candidatos...")
        
        pairs = [[query, c['document']] for c in candidates]
        scores = self.reranker.predict(pairs)
        
        # Adjuntar score y ordenar
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(scores[i])
            
        # Ordenar por rerank_score descendente
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        # Debug Log de cambios
        # for i, r in enumerate(reranked[:3]):
        #    logger.info(f"   #{i+1} Score: {r['rerank_score']:.4f} | {r['metadata'].get('source')}")
            
        return reranked[:top_k]

