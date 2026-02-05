import os
import logging
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import json

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Configurar LLM para RAGAS (Usando Ollama local)
# RAGAS usa LangChain LLMs.
print("🤖 Configurando RAGAS con Ollama local (llama3.2)...")
print("   (Esto puede ser inestable con modelos pequeños. Si falla, intenta ejecutarlo de nuevo)")

# Aumentamos el timeout para evitar errores de espera con Ollama
llm_judge = ChatOllama(
    model="llama3.2",
    temperature=0, # Determinístico para evitar errores de JSON
    timeout=120.0, # 2 minutos por llamada (doble de lo normal)
    num_ctx=4096   # Ventana de contexto amplia
)

embeddings_judge = OllamaEmbeddings(model="llama3.2")

def load_dataset(path="data/golden_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)

def generate_responses(dataset):
    """
    Genera respuestas usando el 'main.py' actual.
    Importante: Importar `app_graph` o la función `chat`.
    """
    from main import app_graph
    
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"🔄 Generando respuestas para {len(dataset)} preguntas...")
    
    for i, item in enumerate(dataset):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"   [{i+1}/{len(dataset)}] Pregunta: {q}")
        
        # Invocar Grafo
        initial_state = {
            "pregunta": q,
            "style": "Formal",
            "debug_pipeline": []
        }
        res = app_graph.invoke(initial_state)
        
        answer = res.get("respuesta", "")
        # Extraer contextos (docs_recuperados es un string gigante, hay que ver si lo podemos trocear o lo pasamos entero)
        # RAGAS espera 'contexts' como lista de strings.
        context_str = res.get("docs_recuperados", "")
        contexts = [context_str] if context_str else [""]
        
        ragas_data["question"].append(q)
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(gt)
        
    return ragas_data

def main():
    print("📋 Iniciando Evaluación RAGAS...")
    
    dataset_raw = load_dataset()
    data_dict = generate_responses(dataset_raw)
    
    # Crear HF Dataset
    eval_dataset = Dataset.from_dict(data_dict)
    
    print("\n🚀 Ejecutando métricas RAGAS (Faithfulness + Answer Relevancy)...")
    print("   (Esto puede tardar varios minutos con Ollama local)")
    
    # Evaluar con manejo de errores y reintentos para modelos locales
    from ragas.run_config import RunConfig
    
    my_run_config = RunConfig(
        timeout=180,       # 3 minutos global por hilo
        max_retries=3,     # Reintentar 3 veces si falla el JSON
        max_wait=60        # Espera entre reintentos
    )
    
    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm_judge,
        embeddings=embeddings_judge,
        run_config=my_run_config,
        raise_exceptions=False # Importante: No parar si falla 1 pregunta
    )
    
    print("\n\n📊 RESULTADOS RAGAS:")
    print("="*60)
    print(results)
    print("="*60)
    
    # Guardar CSV detallado
    df = results.to_pandas()
    df.to_csv("ragas_metrics.csv", index=False)
    print("💾 Resultados guardados en 'ragas_metrics.csv'")

if __name__ == "__main__":
    main()
