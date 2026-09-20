import logging
import os
import requests
import asyncio
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# Cargar variables de entorno
load_dotenv()

# Configuración
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
API_URL = os.getenv("RAG_CHAT_URL", "http://localhost:8000/chat")

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /start"""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="👋 ¡Hola! Soy tu Asistente RAG Multimodal.\n\n"
             "Pregúntame lo que quieras sobre el BOE, convenios o documentación interna.\n"
             "También puedo mostrarte imágenes si las encuentro relevantes."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja el comando /help"""
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="💡 *AyudaRAG*:\n\n"
             "- Simplemente escribe tu duda.\n"
             "- Intentaré responderte con base en los documentos indexados.\n"
             "- Si hay tablas o gráficos, te enviaré las imágenes.",
        parse_mode=ParseMode.MARKDOWN
    )

import base64
from io import BytesIO

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Maneja mensajes de texto e IMÁGENES y consulta la API RAG"""
    chat_id = update.effective_chat.id
    
    # Variables iniciales
    user_text = None
    user_image_b64 = None
    
    # 1. Detectar si es Texto o Foto
    if update.message.text:
        user_text = update.message.text
    elif update.message.photo:
        # Es una foto. Telegram envía varias resoluciones, cogemos la última (mayor calidad)
        photo_file = await update.message.photo[-1].get_file()
        
        # Descargar a memoria
        buffer = BytesIO()
        await photo_file.download_to_memory(buffer)
        buffer.seek(0)
        
        # Convertir a Base64
        user_image_b64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Usar el caption como pregunta, o un default si no hay
        user_text = update.message.caption if update.message.caption else "Describe esta imagen y su contexto."

    if not user_text and not user_image_b64:
        # Ni texto ni imagen
        return

    # Notificar que está "escribiendo..." (u "observando...")
    action = "upload_photo" if user_image_b64 else "typing"
    await context.bot.send_chat_action(chat_id=chat_id, action=action)

    try:
        # Llamada a la API local (Backend)
        payload = {
            "question": user_text,
            "session_id": str(chat_id),
            "style": "Formal",
            "image": user_image_b64 # Añadimos la imagen si existe
        }
        
        # Ejecutar request en un thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(API_URL, json=payload, timeout=120),
        )
        
        if response.status_code == 200:
            data = response.json()
            answer_text = data.get("respuesta", "No tengo respuesta.")
            sources = data.get("sources", [])
            images = data.get("imagenes_finales", [])
            
            # Formatear Fuentes
            sources_text = ""
            if sources:
                sources_text = "\n\n📚 *Fuentes:*\n"
                for i, src in enumerate(sources[:3]):
                    doc_name = src.get('source', 'Doc')
                    page = src.get('page', '?')
                    sources_text += f"- `{doc_name}` (Pág. {page})\n"

            final_msg = answer_text + sources_text
            
            # Enviar chunks largos
            if len(final_msg) > 4000:
                for x in range(0, len(final_msg), 4000):
                    await context.bot.send_message(chat_id=chat_id, text=final_msg[x:x+4000], parse_mode=ParseMode.MARKDOWN)
            else:
                await context.bot.send_message(chat_id=chat_id, text=final_msg, parse_mode=ParseMode.MARKDOWN)

            # Enviar Imágenes generadas/recuperadas
            if images:
                for img_path in images:
                    abs_path = os.path.abspath(img_path)
                    if os.path.exists(abs_path):
                        await context.bot.send_photo(chat_id=chat_id, photo=open(abs_path, 'rb'))
                    else:
                        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ No pude cargar la imagen: {img_path}")

        else:
            await context.bot.send_message(chat_id=chat_id, text=f"⚠️ Error del servidor: {response.text}")

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error de conexión: {str(e)}")

def main():
    if not TELEGRAM_TOKEN:
        print("❌ Error: TELEGRAM_TOKEN no encontrado en variables de entorno o .env")
        return

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    # Aceptamos Texto Y Fotos
    application.add_handler(MessageHandler((filters.TEXT | filters.PHOTO) & (~filters.COMMAND), handle_message))

    print("🤖 Bot de Telegram Iniciado (Soporte Multimodal activado)...")
    application.run_polling()

if __name__ == '__main__':
    main()
