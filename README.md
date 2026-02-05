#  RAG Multimodal "Table-Master" (BOE Edition)

Este proyecto es un **Sistema de RAG (Retrieval-Augmented Generation) Multimodal Avanzado** diseñado para consultar documentos legales oficiales (como el BOE), entender tablas complejas, analizar imágenes y responder preguntas sobre datos de recursos humanos.

## 🚀 Características Principales

### 1. 📄 Ingesta de Documentos Inteligente
*   **Semantic Chunking**: Utiliza embeddings para detectar cambios de tema y cortar el texto de forma lógica, no por caracteres arbitrarios.
*   **Layout Aware (PyMuPDF4LLM)**: Convierte PDFs a Markdown limpio, conservando **Tablas** y estructura visual antes de procesar.
*   **Categorización por Carpetas**: Detecta la estructura de directorios en `docs/` (ej: `Legal/Nóminas`) e inyecta esa categoría en el contexto semántico.
*   **Ingesta Incremental**: Detecta si un archivo ya existe en la base de datos para evitar re-procesarlo (ahorro de tiempo y costes).

### 2. 👁️ Capacidades Multimodales (Vision)
*   **Análisis Visual de Documentos**: Si el documento contiene imágenes o gráficos, el sistema las busca mediante descripción semántica.
*   **Visual Filter (LLaVA)**: Un nodo agente utiliza el modelo de visión `llava` para "mirar" la imagen candidata y verificar si contiene la respuesta exacta (ej: leer un dato numérico de una tabla escaneada).
*   **Query-by-Image**: ¡Nuevo! Puedes subir una foto (nómina, contrato) al chat y preguntar sobre ella. El sistema la analiza con LLaVA y usa esa información para buscar en la base de datos.

### 3. 🧠 Router & Agentes ("Cerebro")
El sistema no busca ciegamente. Tiene un **Router Inteligente** que clasifica tu pregunta:
*   **Ruta "RAG"**: Si preguntas sobre leyes o documentos ("¿Qué dice el artículo 5?"), busca en los PDFs.
*   **Ruta "DATA"**: Si preguntas sobre empleados ("¿Cuántas vacaciones le quedan a Adrian?"), consulta una **base de datos estructurada** (`employees.csv`) usando Pandas.

### 4. 🔍 Técnicas de Recuperación Avanzadas
El sistema implementa 4 técnicas sofisticadas para asegurar que siempre se encuentra el documento más relevante:
*   **Recuperación Híbrida (BM25 + Vector)**: Combina la búsqueda semántica (vectores) con la búsqueda por palabras clave (BM25) para capturar tanto el sentido conceptual como términos exactos (ej. número de artículo).
*   **Cross-Encoder Re-ranking**: Un modelo dedicado (`BAAI/bge-reranker`) re-examina los mejores candidatos de la búsqueda inicial y los reordena meticulosamente por relevancia.
*   **Reciprocal Rank Fusion (RRF)**: Algoritmo que fusiona los resultados de BM25 y Vectores de forma justa y ponderada.
*   **Routing Semántico**: Clasificadores automáticos dirigen la pregunta al subsistema experto adecuado (Data vs Documentos).
---

## 🛠️ Requisitos e Instalación

### 1. Entorno Python
```bash
# Crear entorno (recomendado)
conda create -n Tartanga python=3.11
conda activate Tartanga

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Modelos Locales (Ollama)
Necesitas tener [Ollama](https://ollama.com/) instalado y descargados los siguientes modelos:
```bash
ollama pull llama3       # Cerebro de texto
ollama pull llava        # Visión multimodal
```

### 3. Base de datos
No necesitas instalar nada extra. El proyecto usa **ChromaDB** en modo local (carpeta `chroma_db`).

---

## ▶️ Uso del Sistema

### 1. Ingesta de Datos (Preparación)
Antes de chatear, el sistema necesita aprender. Coloca tus PDFs en la carpeta `docs/` (puedes crear subcarpetas).

```bash
# Ejecutar ingesta inteligente
python ingest_multimodal.py
```
*Este proceso leerá tus PDFs, extraerá tablas y texto, creará chunks semánticos y los guardará en ChromaDB.*

### 2. Iniciar el Backend (Cerebro)
En una terminal:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
*Si ves "🧠 RAG Table-Master Iniciado", todo está bien.*

### 3. Iniciar el Frontend (Chat)
En **otra** terminal:
```bash
streamlit run frontend.py
```
Se abrirá tu navegador en `http://localhost:8501`.

---

## 🧪 Ejemplos de Pruebas

### 🔍 Preguntas RAG (Documentos)
> *"¿Qué dice el BOE sobre las bajas por maternidad?"*
> *"Resume el artículo 14 del convenio."*

### 📊 Preguntas DATA (RRHH)
> *"¿Cuántos días de vacaciones le quedan a Adrian?"*
> *"¿Quién es el HR Manager?"*
*(El sistema detectará que es un dato personal y consultará el CSV automáticamente)*

### 📷 Preguntas con Imagen
1. Abre el desplegable **"📷 Adjuntar imagen"** en el chat.
2. Sube una foto de una tabla o documento.
3. Pregunta: *"¿Es correcta esta nómina según el convenio?"*.
*(El sistema "leerá" tu foto y cruzará la información con los PDFs del BOE)*

---

## 📈 Evaluación del Sistema (Rúbrica SAA)

El proyecto incluye un sistema completo de evaluación cuantitativa para medir la calidad del RAG.

### 1. Evaluación del Buscador (Retrieval)
Script: `eval_retrieval.py`
*   **Métricas**: Hit Rate @ K y MRR (Mean Reciprocal Rank).
*   **Resultados Actuales (v1.5)**:
    | Configuración | Hit Rate | MRR |
    | :--- | :--- | :--- |
    | **Top-3 (Strict)** | **0.80** | **0.70** |
    | **Top-10 (Broad)** | **1.00** | **0.74** |
*   **Ejecución**:
    ```bash
    python eval_retrieval.py
    ```

### 2. Evaluación de Generación (RAGAS)
Script: `eval_ragas.py`
*   **Métricas**: Faithfulness (Fidelidad) y Answer Relevancy.
*   **Resultados Preliminares (Sample n=3)**:
    | Métrica | Puntuación | Descripción |
    | :--- | :--- | :--- |
    | **Faithfulness** | **0.88** | Precisión factual respecto al contexto |
    | **Answer Relevancy** | **0.71** | Relevancia de la respuesta a la pregunta |
*   **Juez**: Utiliza LLM local (Ollama) para evaluar las respuestas generadas sin coste de API.
*   **Dataset**: Utiliza `data/golden_dataset.json` como "Golden Set" de verdad terreno.
*   **Ejecución**:
    ```bash
    python eval_ragas.py
    ```

---

## 📂 Estructura de Proyecto

*   `main.py`: **API Backend**. Contiene el grafo LangGraph, Nodos (Router, Retriever, Vision), y endpoints.
*   `frontend.py`: **Interfaz Streamlit**. Chatbot con soporte de subida de archivos e imágenes.
*   `ingest_multimodal.py`: **Script de Ingesta Avanzada**. PyMuPDF4LLM + Semantic Chunking.
*   `tools_data.py`: Herramienta para consultar `data/employees.csv`.
*   `eval_retrieval.py`: **Script de Validación SAA**. Mide Hit Rate y MRR.
*   `eval_ragas.py`: **Script de Validación SAA**. Mide métricas RAGAS.
*   `data/`: Carpeta para bases de datos estructuradas (CSV) y **Golden Dataset**.
*   `docs/`: Carpeta donde dejas tus PDFs.
*   `chroma_db/`: Base de datos vectorial (generada automáticamente).

---
