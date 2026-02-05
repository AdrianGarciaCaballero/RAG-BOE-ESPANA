import os
import fitz
import pymupdf4llm 
import chromadb
from chromadb.utils import embedding_functions
import shutil
from typing import List, Tuple
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing
import time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_multimodal"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_func = None

def get_embedding_func():
    global _embedding_func
    if _embedding_func is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("⚠️ CUDA no encontrado. Usando CPU (Lento).")
        _embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME,
            device=device
        )
    return _embedding_func

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def semantic_text_splitter(text: str, threshold_percentile=80, max_chunk_size=1500) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) < 3:
        return [text]

    emb_fn = get_embedding_func()
    embeddings = emb_fn(sentences)
    embeddings = np.array(embeddings)
    
    dists = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        dists.append(sim)
    
    if not dists:
        return [text]
        
    threshold = np.percentile(dists, 100 - threshold_percentile)
    
    chunks = []
    current_chunk = [sentences[0]]
    current_len = len(sentences[0])
    
    for i in range(len(dists)):
        sim = dists[i]
        next_sentence = sentences[i+1]
        
        if sim < threshold or (current_len + len(next_sentence) > max_chunk_size):
            chunks.append(" ".join(current_chunk))
            current_chunk = [next_sentence]
            current_len = len(next_sentence)
        else:
            current_chunk.append(next_sentence)
            current_len += len(next_sentence)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks



def fallback_extract_pdf(filepath: str) -> List[dict]:
    doc = fitz.open(filepath)
    data = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            data.append({"text": text, "metadata": {"page": i}})
    return data

def process_file_worker(filepath: str) -> Tuple[str, List[dict], List[dict], List[str]]:
    filename_ext = os.path.basename(filepath)
    filename_base = os.path.splitext(filename_ext)[0]
    
    rel_path = os.path.relpath(filepath, DOCS_DIR)
    raw_cat = os.path.dirname(rel_path)
    category = raw_cat.replace(os.sep, " > ") if raw_cat else "General"

    documents = []
    metadatas = []
    ids = []

    # Signal removed for Windows compatibility
    data = []
    used_strategy = "pymupdf4llm_semantic"
    
    try:
        data = pymupdf4llm.to_markdown(filepath, page_chunks=True)
    except Exception:
        used_strategy = "fallback_standard"
        try:
            data = fallback_extract_pdf(filepath)
        except:
            pass
            
    if not data:
        return filepath, [], [], []

    chunk_counter = 0
    for page_data in data:
        page_num = page_data['metadata']['page']
        content = page_data['text']
        if not content.strip():
            continue
            
        chunks = semantic_text_splitter(content)
        for chunk_text in chunks:
            rich_chunk_text = f"CONTEXTO: Categoría '{category}' | Documento '{filename_base}'\nPÁGINA {page_num+1}:\n{chunk_text}"
            meta = {
                "source": filename_ext,
                "page": page_num + 1,
                "strategy": used_strategy,
                "category": category
            }
            chunk_id = f"{filename_base}_p{page_num}_c{chunk_counter}"
            
            documents.append(rich_chunk_text)
            metadatas.append(meta)
            ids.append(chunk_id)
            chunk_counter += 1
            
    return filepath, documents, metadatas, ids

def process_pdf(filepath: str) -> bool:
    try:
        _, docs, metas, ids = process_file_worker(filepath)
        if docs:
            client = chromadb.PersistentClient(path=CHROMA_PATH)
            emb_fn = get_embedding_func()
            collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)
            collection.add(documents=docs, metadatas=metas, ids=ids)
            return True
        return False
    except Exception as e:
        print(f"Error en process_pdf single: {e}")
        return False

def main():
    if not os.path.exists(DOCS_DIR):
        print(f"⚠️ {DOCS_DIR} no encontrado.")
        return

    os.makedirs(CHROMA_PATH, exist_ok=True)
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    emb_fn = get_embedding_func() 
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    # 1. Recuperar fuentes existentes para RESUME
    existing_completed = set()
    try:
        print("🔍 Verificando historial para RESUMIR ingesta...")
        # Get only metadatas to be faster
        all_data = collection.get(include=["metadatas"])
        for meta in all_data["metadatas"]:
            if meta and "source" in meta:
                existing_completed.add(meta["source"])
        print(f"🔄 Modo RESUME: Se encontraron {len(existing_completed)} documentos ya indexados (se saltarán).")
    except Exception as e:
        print(f"⚠️ No se pudo verificar historial: {e}")

    pdf_files = []
    for root, dirs, files in os.walk(DOCS_DIR):
        for file in files:
            if file.lower().endswith(".pdf"):
                if file in existing_completed:
                    continue
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print("⚠️ No hay PDFs.")
        return
        
    pdf_files.sort()
    total_files = len(pdf_files)
    print(f"📚 Encontrados {total_files} documentos. Iniciando procesamiento PARALELO...")
    num_workers = 2 # Reduced for GPU VRAM safety
    print(f"🚀 Usando {num_workers} procesos (Optimizado para GPU).")

    # Sequential execution for Windows/CUDA stability
    print("⚡ Procesando en modo SECUENCIAL (Mejor para GPU/Windows)...")
    
    with tqdm(total=total_files, desc="⚡ Procesando Archivos", unit="doc") as pbar:
        for filepath in pdf_files:
            # Process directly
            try:
                _, docs, metas, chunk_ids = process_file_worker(filepath)
                filename = os.path.basename(filepath)
                
                if docs:
                    try:
                        # Batch insertion to avoid ChromaDB/SQLite limit (max ~5461 vars)
                        BATCH_SIZE = 1000
                        total_chunks = len(docs)
                        for i in range(0, total_chunks, BATCH_SIZE):
                            end = min(i + BATCH_SIZE, total_chunks)
                            collection.add(
                                documents=docs[i:end],
                                metadatas=metas[i:end],
                                ids=chunk_ids[i:end]
                            )
                    except Exception as e:
                        tqdm.write(f"❌ Error insertando {filename}: {e}")
                else:
                    tqdm.write(f"⚠️ {filename} no generó contenido (ni Fallback).")
            except Exception as e:
                tqdm.write(f"❌ Error fatal procesando {filepath}: {e}")
            
            pbar.update(1)
                
    print("\n✅ Ingesta PDF Completada.")
    
    # 2. Ingesta de Datos CSV (Nuevo)
    try:
        from ingest_csv import ingest_csvs
        ingest_csvs()
    except Exception as e:
        print(f"⚠️ Error ingestando CSVs: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
