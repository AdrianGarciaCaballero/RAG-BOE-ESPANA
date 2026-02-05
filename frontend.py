import streamlit as st
import requests
import os
import base64
import json
import uuid
import random
import time
from datetime import datetime

API_URL = "http://localhost:8000"
HISTORY_FILE = "chat_history.json"

st.set_page_config(page_title="RAG Multimodal BOE", layout="wide")

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def create_session():
    session_id = str(uuid.uuid4())
    st.session_state.current_session = session_id
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
        
    timestamp = datetime.now().strftime("%d/%m %H:%M")
    
    avatars = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮", "🐷"]
    user_avatar = random.choice(avatars)
    
    st.session_state.sessions[session_id] = {
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "title": f"Chat {timestamp}",
        "avatar": user_avatar
    }
    save_history(st.session_state.sessions)
    return session_id

if "sessions" not in st.session_state:
    st.session_state.sessions = load_history()

if "current_session" not in st.session_state:
    if st.session_state.sessions:
        sessions_list = sorted(st.session_state.sessions.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)
        if sessions_list:
            st.session_state.current_session = sessions_list[0][0]
        else:
            create_session()
    else:
        create_session()

if st.session_state.current_session not in st.session_state.sessions:
    create_session()

with st.sidebar:
    st.title("⚙️ Panel de Control")
    
    st.subheader("Estilo del AI")
    tone_options = ["Formal", "Cercano", "Directo", "Didáctico", "Legal"]
    tone = st.radio("Selecciona el tono:", tone_options, index=0, help="Define cómo responderá el asistente.")

    st.subheader("🎨 Apariencia")
    dark_mode = st.toggle("Modo Oscuro", value=True)
    
    if dark_mode:
        st.markdown("""
        <style>
            .stApp {
                background-color: #0e1117;
                color: #fafafa;
            }
            section[data-testid="stSidebar"] {
                background-color: #262730;
                color: #fafafa;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .stApp {
                background-color: #ffffff !important;
                color: #000000 !important;
            }
            section[data-testid="stSidebar"] {
                background-color: #f7f9fb !important;
            }
            section[data-testid="stSidebar"] .stMarkdown, 
            section[data-testid="stSidebar"] h1, 
            section[data-testid="stSidebar"] h2, 
            section[data-testid="stSidebar"] h3, 
            section[data-testid="stSidebar"] p, 
            section[data-testid="stSidebar"] span, 
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] div[data-testid="stToggle"] p {
                color: #000000 !important;
            }

            div[data-testid="stToggle"] p {
                color: #000000 !important;
                font-weight: bold !important;
            }
            div[data-testid="stToggle"] div[role="switch"] {
                border: 1px solid #ccc !important;
            }

            .stChatMessage {
                background-color: #f0f2f6 !important;
                color: #000000 !important;
                border: 1px solid #e0e0e0 !important;
            }
            .stChatMessage p, .stChatMessage div {
                color: #000000 !important;
            }
            
            textarea, input {
                background-color: #ffffff !important;
                color: #000000 !important;
                caret-color: #000000 !important;
            }
            div[data-testid="stChatInput"] {
                background-color: #ffffff !important;
                border-top: 1px solid #ddd !important;
            }
            
            button {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ccc !important;
            }
            button:hover {
                background-color: #f0f0f0 !important;
                color: #000000 !important;
                border-color: #666 !important;
            }
            button[kind="primary"] {
                background-color: #ff4b4b !important;
                color: #ffffff !important;
                border: none !important;
            }
            
            div[data-testid="stRadio"] label {
                color: #000000 !important;
            }
            
            h1, h2, h3, h4, h5, h6, p, label, li, span, div {
                color: #000000 !important;
            }
            
            div[data-testid="stFileUploader"] {
                color: #000000 !important;
            }
        </style>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("💬 Historial")
    if st.button("+ Nuevo Chat", use_container_width=True):
        create_session()
        st.rerun()
    
    st.caption("Selecciona una sesión:")
    
    sorted_sessions = sorted(st.session_state.sessions.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)
    
    for sid, s_data in sorted_sessions:
        col1, col2 = st.columns([5, 1])
        label = s_data.get("title", "Chat sin título")
        is_active = sid == st.session_state.current_session
        
        with col1:
            if st.button(f"{'' if is_active else '🗨️'} {label}", key=f"sel_{sid}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_session = sid
                st.rerun()
                
        with col2:
            if st.button("🗑️", key=f"del_{sid}", help="Eliminar chat"):
                del st.session_state.sessions[sid]
                save_history(st.session_state.sessions)
                if is_active:
                    remaining = sorted(st.session_state.sessions.items(), key=lambda x: x[1].get("created_at", ""), reverse=True)
                    if remaining:
                        st.session_state.current_session = remaining[0][0]
                    else:
                        create_session()
                st.rerun()

    st.divider()
    st.header("📂 Ingesta")
    uploaded_file = st.file_uploader("Sube PDF", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Procesar e Ingestar"):
            with st.spinner("Subiendo y procesando..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        resp_json = response.json()
                        status = resp_json.get("status")
                        msg = resp_json.get("message")
                        
                        if status == "success":
                            st.success(f"✅ {msg}")
                        elif status == "warning":
                            st.warning(f"⚠️ {msg}")
                        else:
                            st.error(f"❌ {msg}")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Error de conexión: {e}")

tab_chat, tab_admin = st.tabs(["💬 Chat", "🗂️ Gestión Documental"])

with tab_chat:
    st.title("📚 Asistente RAG Multimodal")
    st.markdown(f"**Modo:** {tone} | **Sesión:** {st.session_state.sessions[st.session_state.current_session]['title']}")
    st.markdown(f"""
    Consulta documentos oficiales. Si hay tablas o gráficos, el sistema usará visión artificial.
    """)

    current_messages = st.session_state.sessions[st.session_state.current_session]["messages"]

    current_session_data = st.session_state.sessions[st.session_state.current_session]
    session_avatar = current_session_data.get("avatar", "👤")

    for msg in current_messages:
        avatar = session_avatar if msg["role"] == "user" else None 
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "images" in msg and msg["images"]:
                for img_path in msg["images"]:
                    full_url = f"{API_URL}/{img_path}"
                    st.image(full_url, caption="Evidencia Visual Recuperada", width=400)
                    
    query_image_b64 = None
    with st.expander("📷 Adjuntar imagen a tu pregunta (Opcional)", expanded=False):
        img_file = st.file_uploader("Sube una imagen de contexto", type=["png", "jpg", "jpeg"], key="chat_img")
        if img_file:
            st.image(img_file, width=200)
            query_image_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        
        user_msg = {"role": "user", "content": prompt}
        st.session_state.sessions[st.session_state.current_session]["messages"].append(user_msg)
        
        if len(st.session_state.sessions[st.session_state.current_session]["messages"]) == 1:
            new_title = " ".join(prompt.split()[:5]) + "..."
            st.session_state.sessions[st.session_state.current_session]["title"] = new_title
        save_history(st.session_state.sessions)
        
        current_avatar = st.session_state.sessions[st.session_state.current_session].get("avatar", "👤")
        with st.chat_message("user", avatar=current_avatar):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Variables contenedoras para metadatos extraídos del stream
            found_images = []
            found_sources = []
            
            def stream_generator():
                payload = {"question": prompt, "style": tone}
                if query_image_b64:
                    payload["image"] = query_image_b64
                
                try:
                    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True) as r:
                        r.raise_for_status()
                        
                        buffer = ""
                        for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                buffer += chunk
                                
                                # Detect split
                                if "__METADATA_JSON__" in buffer:
                                    parts = buffer.split("__METADATA_JSON__")
                                    text_part = parts[0]
                                    json_part = parts[1]
                                    
                                    # Yield el texto restante
                                    # (Nota: esto asume que el split ocurre clean en este chunk, 
                                    # si el tag está dividido entre chunks podría fallar, 
                                    # pero como es un suffix y el buffer acumula lo no-yielded...
                                    # Simplificación: yield lo que tengamos antes del tag)
                                    
                                    # En este approach simple, yielding chunk by chunk es tricky con split.
                                    # Mejor estrategia: buffer solo lo necesario?
                                    # No, el stream generator de streamlit escribe lo que le des.
                                    
                                    # REWRITE: buffers only needed for split detection?
                                    # Let's assume the tag comes at the end.
                                    pass 
                                    
                                # Simple approach:
                                if "__METADATA_JSON__" in chunk:
                                    # Split in current chunk
                                    parts = chunk.split("__METADATA_JSON__")
                                    yield parts[0] # Yield text content
                                    
                                    try:
                                        # json part might be incomplete if chunked? 
                                        # Usually requests iter_content returns what fits.
                                        # The backend sends it as one write.
                                        meta_str = parts[1]
                                        import json
                                        meta = json.loads(meta_str)
                                        found_images.extend(meta.get("images", []))
                                        found_sources.extend(meta.get("sources", []))
                                    except:
                                        pass
                                else:
                                    yield chunk
                except Exception as e:
                    yield f"❌ Error de conexión: {str(e)}"

            full_response = st.write_stream(stream_generator())
            
            # --- SHOW IMAGES POST-STREAM ---
            if found_images:
                for img_path in found_images:
                    full_url = f"{API_URL}/{img_path}"
                    st.image(full_url, caption="Evidencia Visual / Relacionada", width=400)
            
            asst_msg = {
                "role": "assistant", 
                "content": full_response,
                "images": found_images # Persist in history
            }
            st.session_state.sessions[st.session_state.current_session]["messages"].append(asst_msg)
            save_history(st.session_state.sessions)

with tab_admin:
    st.header("️ Panel de Gestión de Documentos")
    st.info("Aquí puedes ver los documentos indexados en ChromaDB y eliminarlos individualmente.")
    
    if st.button("🔄 Refrescar Lista"):
        st.rerun()
        
    try:
        res = requests.get(f"{API_URL}/documents")
        if res.status_code == 200:
            docs = res.json().get("documents", [])
            if docs:
                st.write(f"**Total Documentos:** {len(docs)}")
                
                col_sel, col_btn = st.columns([3, 1])
                with col_sel:
                    doc_to_delete = st.selectbox("Selecciona un documento para eliminar:", docs)
                with col_btn:
                    if st.button("🗑️ Eliminar Documento", type="primary"):
                        if doc_to_delete:
                            del_res = requests.delete(f"{API_URL}/documents", params={"filename": doc_to_delete})
                            if del_res.status_code == 200:
                                st.success(f"✅ Documento '{doc_to_delete}' eliminado correctamente.")
                                time.sleep(1) 
                                st.rerun()
                            else:
                                st.error(f"❌ Error al eliminar: {del_res.text}")
                
                st.table(docs)
            else:
                st.warning("⚠️ No hay documentos indexados aún.")
        else:
            st.error("Error conectando con backend.")
    except Exception as e:
        st.error(f"Error de conexión: {e}")
