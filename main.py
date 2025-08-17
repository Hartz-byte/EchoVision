"""
Main FastAPI application for AI Chatbot with Image Generation
"""
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import io
import base64
import uuid
from typing import Optional
import logging

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm.mistral_handler import MistralHandler
from core.image_generation.stable_diffusion import ImageGenerator
from core.utils.language_handler import LanguageHandler
from core.llm.memory import ConversationMemory
from config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Initialize components
mistral_handler = None
image_generator = None
language_handler = LanguageHandler()
conversation_memories = {}

app = FastAPI(
    title="AI Chatbot with Image Generation",
    description="A local AI chatbot that supports text conversation and image generation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    image_data: Optional[str] = None
    language: str
    session_id: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global mistral_handler, image_generator
    
    try:
        logger.info("🤖 Loading Mistral model...")
        mistral_handler = MistralHandler(
            model_path=settings.MISTRAL_MODEL_PATH,
            n_gpu_layers=settings.MISTRAL_N_GPU_LAYERS,
            n_ctx=settings.MISTRAL_N_CTX,
            n_threads=settings.MISTRAL_N_THREADS
        )
        logger.info("✅ Mistral model loaded successfully!")
        
        logger.info("🎨 Loading Stable Diffusion model...")
        image_generator = ImageGenerator(settings.SD_MODEL_PATH)
        logger.info("✅ Stable Diffusion model loaded successfully!")
        
        logger.info("🚀 All models loaded successfully! Ready to chat and generate images!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise

@app.get("/")
async def root():
    return {
        "message": "AI Chatbot with Image Generation API", 
        "status": "running",
        "version": "1.0.0",
        "features": ["text_chat", "image_generation", "multilingual_support"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mistral_loaded": mistral_handler is not None,
        "image_gen_loaded": image_generator is not None,
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "gpu_available": "cuda" if image_generator and image_generator.device == "cuda" else "cpu"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"💬 Processing chat request from session: {request.session_id[:8]}...")
        
        # Get or create conversation memory for session
        if request.session_id not in conversation_memories:
            conversation_memories[request.session_id] = ConversationMemory(
                max_tokens=settings.MAX_CONVERSATION_TOKENS
            )
        
        memory = conversation_memories[request.session_id]
        
        # Detect language
        detected_lang = language_handler.detect_language(request.message)
        logger.info(f"🌍 Detected language: {detected_lang}")
        
        # Get conversation history
        history = memory.get_conversation_string()
        
        # Create context-aware prompt
        prompt = language_handler.create_contextual_prompt(
            request.message, history, detected_lang
        )
        
        # Generate response
        logger.info("🧠 Generating response...")
        response = mistral_handler.generate_response(prompt)
        
        # Check for image generation request
        image_data = None
        if "IMAGE_REQUEST:" in response:
            parts = response.split("IMAGE_REQUEST:", 1)
            text_response = parts[0].strip()
            image_prompt = parts[1].strip() if len(parts) > 1 else ""
            
            # Always use text response (remove IMAGE_REQUEST from user-facing response)
            response = text_response
            
            # Validate if this is a legitimate image request
            user_message_lower = request.message.lower()
            image_keywords = [
                'image', 'picture', 'photo', 'generate', 'create', 'draw', 'make', 'show me', 'paint', 'design',
                'छवि', 'तस्वीर', 'फोटो', 'बनाओ', 'दिखाओ', 'चित्र', 'स्केच',
                'imagen', 'foto', 'generar', 'crear', 'hacer', 'mostrar', 'dibujar', 'pintar', 'diseñar',
                'image', 'photo', 'générer', 'créer', 'faire', 'montrer', 'dessiner', 'peindre', 'concevoir'
            ]
            
            # Check if user actually requested an image
            has_image_keyword = any(keyword in user_message_lower for keyword in image_keywords)
            
            if has_image_keyword and image_prompt:
                # Limit prompt length for CLIP (max 77 tokens ≈ 300 characters)
                if len(image_prompt) > 300:
                    truncated = image_prompt[:300]
                    last_space = truncated.rfind(' ')
                    image_prompt = truncated[:last_space] if last_space > 200 else truncated[:300]
                
                try:
                    logger.info(f"🎨 Generating image for: {image_prompt[:50]}...")
                    image = image_generator.generate_image(
                        prompt=image_prompt,
                        num_inference_steps=settings.SD_NUM_INFERENCE_STEPS,
                        guidance_scale=settings.SD_GUIDANCE_SCALE,
                        width=settings.SD_WIDTH,
                        height=settings.SD_HEIGHT
                    )
                    
                    # Convert image to base64
                    buffer = io.BytesIO()
                    image.save(buffer, format="PNG")
                    image_data = base64.b64encode(buffer.getvalue()).decode()
                    
                    logger.info("✅ Image generated successfully!")
                except Exception as e:
                    logger.error(f"❌ Error generating image: {e}")
                    response += f"\n\nSorry, I couldn't generate the image due to an error: {str(e)}"
            else:
                if not has_image_keyword:
                    logger.info("🚫 IMAGE_REQUEST detected but user didn't ask for image - skipping generation")
                elif not image_prompt:
                    logger.warning("⚠️ IMAGE_REQUEST found but no prompt provided - skipping generation")

        
        # Save conversation
        memory.add_message(request.message, response)
        
        logger.info("✅ Chat request processed successfully!")
        
        return ChatResponse(
            response=response,
            image_data=image_data,
            language=detected_lang,
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    if session_id in conversation_memories:
        del conversation_memories[session_id]
        logger.info(f"🗑️ Cleared session: {session_id[:8]}...")
        return {"message": f"Session {session_id} cleared successfully"}
    return {"message": "Session not found"}

@app.get("/stats")
async def get_stats():
    """Get application statistics"""
    return {
        "active_sessions": len(conversation_memories),
        "total_conversations": sum(len(memory.messages) for memory in conversation_memories.values()),
        "supported_languages": settings.SUPPORTED_LANGUAGES,
        "model_info": {
            "llm_model": "Mistral 7B Instruct v0.2",
            "image_model": "Stable Diffusion v1.5",
            "device": image_generator.device if image_generator else "unknown"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🚀 Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        app, 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        log_level="info"
    )
