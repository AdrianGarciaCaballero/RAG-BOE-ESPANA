import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import torch

# Configuracion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "rag_multimodal" # Misma coleccion que los PDFs
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FILES = {
    "employees": os.path.join(DATA_DIR, "employees.csv"),
    "vacations": os.path.join(DATA_DIR, "Tabla Dinámica de VACACIONES.csv"),
    "sick_leave": os.path.join(DATA_DIR, "Tabla de Bajas Médicas.csv")
}

def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        device=device
    )
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)

def ingest_csvs():
    print("📊 Iniciando Ingesta CSV - MODO: 1 CHUNK POR FILA (con contexto completo)...")
    collection = get_chroma_collection()
    
    # Cargar datos base de empleados
    employee_base = {}  # {emp_id: {name, role, vacation_days, etc}}
    if os.path.exists(FILES["employees"]):
        try:
            df = pd.read_csv(FILES["employees"])
            for _, row in df.iterrows():
                emp_id = str(row['id'])
                employee_base[emp_id] = {
                    "name": row['name'],
                    "role": row['role'],
                    "vacation_days_left": row['vacation_days_left'],
                    "last_pay_raise": row['last_pay_raise']
                }
        except Exception as e:
            print(f"   ⚠️ Error leyendo employees.csv: {e}")
    
    documents = []
    metadatas = []
    ids = []
    
    # 1. VACACIONES: 1 chunk por solicitud
    if os.path.exists(FILES["vacations"]):
        try:
            df = pd.read_csv(FILES["vacations"])
            df['ID_Empleado'] = df['ID_Empleado'].astype(str)
            
            for idx, row in df.iterrows():
                emp_id = row['ID_Empleado']
                emp_name = row['Nombre_Empleado']
                
                # Construir chunk con TODA la info
                # Incluir datos base si existen
                base_info = ""
                if emp_id in employee_base:
                    base = employee_base[emp_id]
                    base_info = f"""
EMPLEADO: {base['name']} (ID: {emp_id})
Puesto: {base['role']}
Vacaciones disponibles: {base['vacation_days_left']} días
"""
                
                chunk_text = f"""{base_info}
SOLICITUD DE VACACIONES:
Empleado: {emp_name} (ID: {emp_id})
Departamento: {row['Departamento']}
Tipo de contrato: {row['Tipo_Contrato']}
Periodo: {row['Fecha_Inicio']} a {row['Fecha_Fin']}
Días solicitados: {row['Días_Solicitados']}
Estado: {row['Estado']}
"""
                
                documents.append(chunk_text.strip())
                metadatas.append({
                    "source": "vacaciones_rrhh",
                    "type": "vacation_request",
                    "employee_id": emp_id,
                    "employee_name": emp_name,
                    "status": row['Estado']
                })
                ids.append(f"vac_{emp_id}_{idx}")
                
            print(f"   ✅ Indexadas {len(df)} solicitudes de vacaciones")
        except Exception as e:
            print(f"   ❌ Error en vacaciones: {e}")
    
    # 2. BAJAS: 1 chunk por baja
    if os.path.exists(FILES["sick_leave"]):
        try:
            df = pd.read_csv(FILES["sick_leave"])
            df['ID_Empleado'] = df['ID_Empleado'].astype(str)
            
            for idx, row in df.iterrows():
                emp_id = row['ID_Empleado']
                emp_name = row['Nombre_Empleado']
                
                base_info = ""
                if emp_id in employee_base:
                    base = employee_base[emp_id]
                    base_info = f"""
EMPLEADO: {base['name']} (ID: {emp_id})
Puesto: {base['role']}
"""
                
                chunk_text = f"""{base_info}
BAJA MÉDICA:
Empleado: {emp_name} (ID: {emp_id})
ID Baja: {row['ID_Baja']}
Tipo: {row['Tipo_Baja']}
Motivo: {row['Motivo_Detallado']}
Periodo: {row['Fecha_Inicio']} a {row['Fecha_Alta']}
Días totales: {row['Dias_Totales']}
Coste estimado: {row['Coste_Empresa_Est']}
"""
                
                documents.append(chunk_text.strip())
                metadatas.append({
                    "source": "bajas_rrhh",
                    "type": "sick_leave",
                    "employee_id": emp_id,
                    "employee_name": emp_name,
                    "sick_type": row['Tipo_Baja']
                })
                ids.append(f"sick_{row['ID_Baja']}")
                
            print(f"   ✅ Indexadas {len(df)} bajas médicas")
        except Exception as e:
            print(f"   ❌ Error en bajas: {e}")
    
    # 3. FICHAS BASE (solo empleados que NO tienen vacaciones ni bajas ya indexadas)
    for emp_id, data in employee_base.items():
        # Crear chunk de ficha base
        chunk_text = f"""
EMPLEADO: {data['name']} (ID: {emp_id})
Puesto: {data['role']}
Vacaciones disponibles: {data['vacation_days_left']} días
Última subida salarial: {data['last_pay_raise']}
"""
        documents.append(chunk_text.strip())
        metadatas.append({
            "source": "employees_rrhh",
            "type": "employee_card",
            "employee_id": emp_id,
            "employee_name": data['name']
        })
        ids.append(f"emp_card_{emp_id}")
    
    print(f"   ✅ Indexadas {len(employee_base)} fichas de empleados")
    
    # Indexar todo CON LOGGING DETALLADO
    if documents:
        print(f"\n📥 Intentando indexar {len(documents)} chunks...")
        print(f"   - Vacaciones: {sum(1 for m in metadatas if m['source'] == 'vacaciones_rrhh')}")
        print(f"   - Bajas: {sum(1 for m in metadatas if m['source'] == 'bajas_rrhh')}")
        print(f"   - Fichas: {sum(1 for m in metadatas if m['source'] == 'employees_rrhh')}")
        
        # DETECCIÓN DE IDs DUPLICADOS
        unique_ids = set(ids)
        if len(unique_ids) != len(ids):
            print(f"\n⚠️ ADVERTENCIA: Hay {len(ids) - len(unique_ids)} IDs duplicados!")
            from collections import Counter
            id_counts = Counter(ids)
            duplicates = [id for id, count in id_counts.items() if count > 1]
            print(f"   IDs duplicados: {duplicates[:5]}...")  # Mostrar primeros 5
        
        # Debug: mostrar primeros 3 IDs
        print(f"\n   Primeros 3 IDs a indexar:")
        for i in range(min(3, len(ids))):
            print(f"     - {ids[i]} ({metadatas[i]['source']})")
        
        # VERIFICAR si ya existen en la DB antes de añadir
        print(f"\n   Verificando IDs existentes en DB...")
        existing = collection.get(ids=ids[:10])  # Probar con los primeros 10
        if existing['ids']:
            print(f"   ⚠️ {len(existing['ids'])} de los primeros 10 IDs YA EXISTEN en la DB")
        else:
           print(f"   ✅ IDs no existen previamente")
        
        try:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"\n✅ TOTAL: {len(documents)} chunks indexados correctamente")
            
            # Verificación post-ingesta
            vac_count = len(collection.get(where={"source": "vacaciones_rrhh"})['ids'])
            sick_count = len(collection.get(where={"source": "bajas_rrhh"})['ids'])
            emp_count = len(collection.get(where={"source": "employees_rrhh"})['ids'])
            print(f"\n🔍 Verificación en ChromaDB:")
            print(f"   - Vacaciones: {vac_count}")
            print(f"   - Bajas: {sick_count}")
            print(f"   - Fichas: {emp_count}")
            
        except Exception as e:
            print(f"\n❌ ERROR CRÍTICO al indexar: {e}")
            print(f"   Tipo de error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Intentar indexar en lotes pequeños para identificar el problema
            print("\n🔧 Intentando indexar en lotes de 10...")
            for i in range(0, len(documents), 10):
                batch_docs = documents[i:i+10]
                batch_meta = metadatas[i:i+10]
                batch_ids = ids[i:i+10]
                try:
                    collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
                    print(f"   ✅ Lote {i//10 + 1} OK ({len(batch_ids)} chunks)")
                except Exception as batch_error:
                    print(f"   ❌ Lote {i//10 + 1} FALLÓ: {batch_error}")
                    print(f"      IDs problemáticos: {batch_ids}")

if __name__ == "__main__":
    ingest_csvs()
