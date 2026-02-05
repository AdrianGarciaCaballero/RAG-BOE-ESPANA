import chromadb
from chromadb.utils import embedding_functions
import torch

# Test de la lógica de recuperación directa
client = chromadb.PersistentClient(path='chroma_db')
device = "cuda" if torch.cuda.is_available() else "cpu"
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2',
    device=device
)
collection = client.get_collection(name='rag_multimodal', embedding_function=ef)

# Simular detección de nombres
question = "¿Sergio Ramos tiene vacaciones pendientes?"
palabras = question.split()
nombres_detectados = [word for word in palabras if word[0].isupper() and len(word) > 2 and word.isalpha()]

print(f"Nombres detectados: {nombres_detectados}\n")

if nombres_detectados:
    # Recuperar TODOS los chunks de RRHH
    all_rrhh = collection.get(
        where={"source": {"$in": ["vacaciones_rrhh", "bajas_rrhh", "employees_rrhh"]}}
    )
   
    print(f"Total chunks RRHH: {len(all_rrhh['ids'])}\n")
    
    # Filtrar por nombre
    direct_employee_docs = []
    for i, meta in enumerate(all_rrhh['metadatas']):
        emp_name = meta.get('employee_name', '')
        if any(nombre.lower() in emp_name.lower() for nombre in nombres_detectados):
            direct_employee_docs.append(all_rrhh['documents'][i])
            print(f"Match #{len(direct_employee_docs)}: {emp_name}")
            if len(direct_employee_docs) <= 3:  # Mostrar primeros 3
                print(f"  Doc: {all_rrhh['documents'][i][:100]}...\n")
    
    print(f"\nTotal docs recuperados directamente: {len(direct_employee_docs)}")
else:
    print("No se detectaron nombres")
