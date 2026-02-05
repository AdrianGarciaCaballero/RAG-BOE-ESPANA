import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import retriever

def test_retrieval():
    state = {
        "pregunta": "Quiero ver el gráfico de crecimiento de ventas",
        "docs_recuperados": "",
        "imagenes_candidatas": [],
        "datos_visuales_extraidos": "",
        "imagenes_finales": [],
        "respuesta": ""
    }
    
    print(f"🔎 Pregunta: {state['pregunta']}")
    result = retriever(state)
    
    candidates = result.get("imagenes_candidatas", [])
    print(f"📸 Candidatas encontradas: {candidates}")
    
    found = any("test_chart.png" in c for c in candidates)
    if found:
        print("✅ ÉXITO: Se encontró la imagen 'test_chart.png'.")
    else:
        print("❌ FALLO: No se encontró la imagen esperada.")

if __name__ == "__main__":
    test_retrieval()
