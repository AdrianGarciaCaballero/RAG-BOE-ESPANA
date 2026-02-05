import os
import json
import chromadb
import torch
from chromadb.utils import embedding_functions
import ollama

# --- CONFIGURACIÓN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LABELED_IMAGES_DIR = os.path.join(STATIC_DIR, "labeled_images")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_images" # Nueva colección para imágenes
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VISION_MODEL = "llama3.2-vision"

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=device
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def resize_image_if_needed(image_path, max_size=1024):
    """Redimensiona la imagen si excede max_size para evitar que Ollama se bloquee."""
    import io
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # Convertir a RGB si es necesario (ej. PNG con transparencia)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
                
            w, h = img.size
            if w <= max_size and h <= max_size:
                return image_path # No cambiar nada

            # Calcular nuevo tamaño
            ratio = min(max_size / w, max_size / h)
            new_size = (int(w * ratio), int(h * ratio))
            
            print(f"   📉 Redimensionando imagen (Large): {w}x{h} -> {new_size}")
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Guardar en buffer de bytes
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            return buf.getvalue() # Retornar bytes
    except ImportError:
        print("   ⚠️ PIL no instalado. Usando imagen original (puede ser lento).")
        return image_path
    except Exception as e:
        print(f"   ⚠️ Error redimensionando: {e}. Usando original.")
        return image_path

def generate_auto_caption(image_path):
    import signal
    
    def handler(signum, frame):
        raise TimeoutError("⏳ Tiempo de espera agotado (60s)")
    
    # Registrar el manejador de la señal
    signal.signal(signal.SIGALRM, handler)
    
    print(f"   🧠 [IA] Generando descripción para: {os.path.basename(image_path)}...")
    print("      (Esto puede tardar unos segundos...)")
    
    try:
        # Pre-procesar imagen (resize)
        image_input = resize_image_if_needed(image_path)
        
        # Iniciar cuenta atrás de 45 segundos
        signal.alarm(45)
        
        res = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Describe esta imagen en detalle para ser usada en un buscador semántico. Céntrate en el contenido visual, texto visible, tipo de gráfico y datos clave. Responde en español.',
                'images': [image_input]
            }],
            options={"num_ctx": 2048, "temperature": 0.2}
        )
        
        # Desactivar alarma si termina bien
        signal.alarm(0)
        
        print("      ✅ Descripción generada con éxito.")
        return res['message']['content'].strip()
        
    except TimeoutError as te:
        print(f"      ⏭️  SALTANDO imagen por bloqueo: {te}")
        return None # Saltamos esta imagen
    except Exception as e:
        signal.alarm(0) # Asegurar apagar alarma
        print(f"   ❌ Error generando caption: {e}")
        return None

def main():
    if not os.path.exists(LABELED_IMAGES_DIR):
        print(f"⚠️ El directorio {LABELED_IMAGES_DIR} no existe. Creándolo...")
        os.makedirs(LABELED_IMAGES_DIR)
        print("ℹ️ Coloca tus imágenes y el archivo labels.json aquí.")
        return

    collection = get_chroma_collection()
    
    # Intenta cargar labels.json
    labels_path = os.path.join(LABELED_IMAGES_DIR, "labels.json")
    labels = {}
    if os.path.exists(labels_path):
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
        except Exception as e:
            print(f"⚠️ Error leyendo labels.json: {e}")
            
    # Listar imágenes
    valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [f for f in os.listdir(LABELED_IMAGES_DIR) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not images:
        print("⚠️ No se encontraron imágenes en static/labeled_images.")
        return

    print(f"🔄 Procesando {len(images)} imágenes...")
    
    # --- MODO INCREMENTAL ---
    # No borramos la colección. Solo cargamos la existente.
    collection = get_chroma_collection()
    
    # Obtener IDs ya indexados para saltarlos
    existing_ids = set()
    try:
        # include=[] solo trae IDs y metadata básica, es más rápido
        existing_data = collection.get(include=[])
        if existing_data and 'ids' in existing_data:
            existing_ids = set(existing_data['ids'])
    except Exception as e:
        print(f"ℹ️  Colección nueva o error leyendo IDs: {e}")

    labels_updated = False
    
    for img_filename in images:
        # 0. Chequeo de duplicados (Saltar si ya existe en DB)
        if img_filename in existing_ids:
            # Opcional: Descomentar para ver logs de saltos
            # print(f"   ⏭️  Saltando (ya indexada): {img_filename}")
            continue

        img_path = os.path.join(LABELED_IMAGES_DIR, img_filename)
        
        # 1. Obtener o Generar Descripción
        if img_filename in labels:
            description = labels[img_filename]
            source = "MANUAL"
        else:
            # AUTO-CAPTIONING
            caption = generate_auto_caption(img_path)
            if caption:
                description = caption
                labels[img_filename] = description
                labels_updated = True
                source = "AUTO-IA"
                
                # --- CHECKPOINT: GUARDADO INCREMENTAL ---
                # Guardamos INMEDIATAMENTE para no perder el progreso si la siguiente imagen falla
                try:
                    with open(labels_path, "w", encoding="utf-8") as f:
                        json.dump(labels, f, indent=4, ensure_ascii=False)
                    print(f"      💾 Checkpoint guardado para: {img_filename}")
                except Exception as e:
                    print(f"      ⚠️ Error guardando checkpoint: {e}")
                # ----------------------------------------
            else:
                description = img_filename # Fallback
                source = "FILENAME"

        # 2. Indexar
        print(f"   [IMG] Indexando ({source}): {img_filename} -> '{description[:40]}...'")
        
        collection.add(
            documents=[description],
            metadatas=[{"filename": img_filename, "type": "image", "source": source}],
            ids=[img_filename]
        )
        
    print(f"✅ Ingesta completada. Total imágenes: {len(images)}")

if __name__ == "__main__":
    main()
