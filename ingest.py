import os
import fitz
import chromadb
from chromadb.utils import embedding_functions
import uuid
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_multimodal"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(CHROMA_PATH, exist_ok=True)

MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 100
MIN_IMAGE_SIZE_BYTES = 2048

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def extract_images_from_page(page, doc, doc_id, page_num) -> List[str]:
    valid_images = []
    image_list = page.get_images(full=True)
    
    for img_idx, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        width = base_image["width"]
        height = base_image["height"]
        ext = base_image["ext"]
        
        if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
            continue
        if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
            continue
            
        filename = f"{doc_id}_p{page_num}_{img_idx}.{ext}"
        filepath = os.path.join(IMAGES_DIR, filename)
        
        with open(filepath, "wb") as f:
            f.write(image_bytes)
            
        print(f"   [IMG] Guardada: {filename} ({width}x{height}px)")
        valid_images.append(f"static/images/{filename}")
        
    return valid_images

def text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks

def process_file(filepath: str):
    filename = os.path.basename(filepath)
    print(f"\nProcesando: {filename}")
    
    doc = fitz.open(filepath)
    collection = get_chroma_collection()
    doc_id = str(uuid.uuid4())[:8]
    
    total_chunks = 0
    total_images = 0
    
    for i, page in enumerate(doc):
        images_on_page = extract_images_from_page(page, doc, doc_id, i)
        total_images += len(images_on_page)
        
        text = page.get_text()
        if not text.strip():
            print(f"   [WARN] Página {i} vacía de texto (posible imagen escaneada).")
            continue
            
        chunks = text_splitter(text)
        
        for j, chunk_text in enumerate(chunks):
            img_metadata_val = ",".join(images_on_page) if images_on_page else "None"
            
            metadata = {
                "source": filename,
                "page": i,
                "chunk_index": j,
                "image_path_list": img_metadata_val
            }
            
            chunk_id = f"{doc_id}_p{i}_c{j}"
            
            collection.add(
                documents=[chunk_text],
                metadatas=[metadata],
                ids=[chunk_id]
            )
            total_chunks += 1
            
    print(f"✅ Finalizado {filename}: {total_chunks} chunks, {total_images} imagénes válidas.")

def main():
    if not os.path.exists(DOCS_DIR):
        print(f"Carpeta {DOCS_DIR} no existe. Creándola...")
        os.makedirs(DOCS_DIR)
        print("Por favor, pon tus PDFs en la carpeta 'docs' y ejecuta de nuevo.")
        return

    files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]
    
    if not files:
        print("No se encontraron PDFs en ./docs/")
        return
        
    print(f"Encontrados {len(files)} documentos.")
    for f in files:
        process_file(os.path.join(DOCS_DIR, f))

if __name__ == "__main__":
    main()
