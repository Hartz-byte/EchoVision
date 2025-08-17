"""
Streamlit Frontend for AI Chatbot with Image Generation
"""
import streamlit as st
import requests
import base64
import io
import uuid
from PIL import Image
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AI Chatbot with Image Generation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .language-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #4caf50;
        color: white;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .status-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .status-online {
        background-color: #4caf50;
        color: white;
    }
    .status-offline {
        background-color: #f44336;
        color: white;
    }
    .stats-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"
    if "model_info" not in st.session_state:
        st.session_state.model_info = {}

def check_api_status():
    """Check if the backend API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy", data
        return False, {}
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False, {}

def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.error(f"API stats failed: {e}")
        return {}

def send_chat_message(message: str):
    """Send message to backend API"""
    try:
        with st.spinner("🤔 Thinking..."):
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={
                    "message": message,
                    "session_id": st.session_state.session_id
                },
                timeout=120  # Longer timeout for image generation
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("⏰ Request timed out. The model might be taking longer to respond.")
        return None
    except Exception as e:
        st.error(f"❌ Error communicating with backend: {e}")
        return None

def clear_conversation():
    """Clear the conversation history"""
    try:
        response = requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
        if response.status_code == 200:
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("🗑️ Conversation cleared!")
            st.rerun()
        else:
            st.error("❌ Failed to clear conversation on server")
    except Exception as e:
        st.error(f"❌ Error clearing conversation: {e}")

def display_chat_message(message_data, is_user=True):
    """Display a chat message with proper styling"""
    role = "user" if is_user else "assistant"
    
    with st.chat_message(role):
        if is_user:
            st.markdown(f'<div class="chat-message user-message">{message_data["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            # Display text response
            content = message_data["content"]
            language = message_data.get("language", "en")
            
            # Language mapping for display
            lang_display = {
                "en": "🇺🇸 EN",
                "hi": "🇮🇳 HI", 
                "es": "🇪🇸 ES",
                "fr": "🇫🇷 FR"
            }
            
            lang_badge = lang_display.get(language, f"🌍 {language.upper()}")
            
            st.markdown(f'<div class="chat-message assistant-message">{content}<span class="language-badge">{lang_badge}</span></div>', 
                       unsafe_allow_html=True)
            
            # Display image if present
            if "image" in message_data and message_data["image"]:
                st.image(
                    message_data["image"], 
                    caption="🎨 Generated Image", 
                    use_column_width=True,
                    output_format="PNG"
                )

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Chatbot with Image Generation</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # API Status Check
        api_healthy, api_data = check_api_status()
        st.session_state.api_status = "online" if api_healthy else "offline"
        
        if api_healthy:
            st.markdown('<div class="status-indicator status-online">✅ API Online</div>', 
                       unsafe_allow_html=True)
            
            if api_data:
                st.success(f"🧠 LLM: {'✅' if api_data.get('mistral_loaded') else '❌'}")
                st.success(f"🎨 Image Gen: {'✅' if api_data.get('image_gen_loaded') else '❌'}")
                st.info(f"💻 GPU: {api_data.get('gpu_available', 'unknown').upper()}")
        else:
            st.markdown('<div class="status-indicator status-offline">❌ API Offline</div>', 
                       unsafe_allow_html=True)
            st.error("Please start the backend server first!")
            st.code("python main.py", language="bash")
        
        st.divider()
        
        # API Statistics
        if api_healthy:
            stats = get_api_stats()
            if stats:
                st.header("📊 Statistics")
                st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                st.metric("Active Sessions", stats.get("active_sessions", 0))
                st.metric("Total Conversations", stats.get("total_conversations", 0))
                
                model_info = stats.get("model_info", {})
                if model_info:
                    st.write(f"**LLM:** {model_info.get('llm_model', 'Unknown')}")
                    st.write(f"**Image Model:** {model_info.get('image_model', 'Unknown')}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Language Information
        st.header("🌍 Language Support")
        st.write("**Supported Languages:**")
        st.write("• 🇺🇸 **English** - Full support")
        st.write("• 🇮🇳 **हिंदी (Hindi)** - पूर्ण समर्थन")
        st.write("• 🇪🇸 **Español** - Soporte completo")  
        st.write("• 🇫🇷 **Français** - Support complet")
        
        st.divider()
        
        # Image Generation Info
        st.header("🎨 Image Generation")
        st.write("**Trigger phrases:**")
        st.write("• 'Create an image of...'")
        st.write("• 'Generate a picture of...'")
        st.write("• 'Draw me...'")
        st.write("• 'Show me an image...'")
        st.write("• Or equivalent in any supported language")
        
        st.divider()
        
        # Session Management
        st.header("💬 Session Info")
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
        
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.success("New session started!")
                st.rerun()
    
    # Main chat interface
    if not api_healthy:
        st.warning("⚠️ **Backend API is not running.** Please start the server first:")
        st.code("python main.py")
        
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            **To start the backend:**
            1. Open a terminal in your project directory
            2. Activate your virtual environment: `source venv/bin/activate`
            3. Run the server: `python main.py`
            4. Wait for "All models loaded successfully!" message
            5. Refresh this page
            """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, is_user=message["role"] == "user")
    
    # Chat input
    if prompt := st.chat_input("Type your message in any supported language... 💬"):
        # Add user message to chat
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        display_chat_message(user_message, is_user=True)
        
        # Send to backend
        response_data = send_chat_message(prompt)
        
        if response_data:
            # Process response
            assistant_message = {
                "role": "assistant",
                "content": response_data["response"],
                "language": response_data.get("language", "en")
            }
            
            # Handle image if present
            if response_data.get("image_data"):
                try:
                    image_bytes = base64.b64decode(response_data["image_data"])
                    image = Image.open(io.BytesIO(image_bytes))
                    assistant_message["image"] = image
                    
                    # Save image to show download option
                    timestamp = int(time.time())
                    filename = f"generated_image_{timestamp}.png"
                    
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    st.error(f"Error displaying image: {e}")
            
            # Add to session state and display
            st.session_state.messages.append(assistant_message)
            display_chat_message(assistant_message, is_user=False)
        else:
            error_message = {
                "role": "assistant",
                "content": "❌ Sorry, I encountered an error while processing your request. Please try again or check if the backend server is running properly.",
                "language": "en"
            }
            st.session_state.messages.append(error_message)
            display_chat_message(error_message, is_user=False)

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 1rem;">' +
        '🚀 Powered by <strong>Mistral 7B</strong> + <strong>Stable Diffusion</strong> | ' +
        '💻 Local AI Chatbot | 🌍 Multi-language Support' +
        '</div>',
        unsafe_allow_html=True
    )
    
    # Debug info (only show in development)
    with st.expander("🐛 Debug Info", expanded=False):
        st.json({
            "session_id": st.session_state.session_id,
            "total_messages": len(st.session_state.messages),
            "api_status": st.session_state.api_status,
            "api_base_url": API_BASE_URL
        })

if __name__ == "__main__":
    main()
