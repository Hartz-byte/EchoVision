"""
EchoVision - Streamlit Frontend for AI Chatbot with Image Generation
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
    page_title="EchoVision - AI Chatbot",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main app styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }

    /* Chat message containers */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 1rem 0;
    }
    
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 1rem 0;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .assistant-bubble {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 2px 10px rgba(245, 87, 108, 0.3);
        position: relative;
    }
    
    .language-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* Status indicators */
    .status-online {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(76, 175, 80, 0.3);
    }
    
    .status-offline {
        background: linear-gradient(45deg, #f44336, #d32f2f);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.3);
    }

    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .sidebar-section h3 {
        margin-top: 0;
        color: white;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }

    /* Generated image styling */
    .generated-image {
        max-width: 300px;
        max-height: 300px;
        width: auto;
        height: auto;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    
    .image-caption {
        text-align: center;
        font-style: italic;
        color: rgba(255, 255, 255, 0.8);
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }

    /* Animation for messages */
    .message-animation {
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom spinner */
    .creating-spinner {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.1);
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
    if "total_messages" not in st.session_state:
        st.session_state.total_messages = 0
    if "images_generated" not in st.session_state:
        st.session_state.images_generated = 0

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

def is_image_request(message: str):
    """Check if the message is an image generation request"""
    image_keywords = [
        'image', 'picture', 'photo', 'generate', 'create', 'draw', 'make', 'show me', 'paint', 'design',
        'छवि', 'तस्वीर', 'फोटो', 'बनाओ', 'दिखाओ', 'चित्र', 'स्केच',
        'imagen', 'foto', 'generar', 'crear', 'hacer', 'mostrar', 'dibujar', 'pintar', 'diseñar',
        'image', 'photo', 'générer', 'créer', 'faire', 'montrer', 'dessiner', 'peindre', 'concevoir'
    ]
    return any(keyword in message.lower() for keyword in image_keywords)

def send_chat_message(message: str):
    """Send message to backend API"""
    try:
        # Check if it's an image request to show appropriate spinner
        if is_image_request(message):
            with st.spinner("🎨 Creating your image..."):
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "message": message,
                        "session_id": st.session_state.session_id
                    },
                    timeout=120
                )
        else:
            with st.spinner("🤔 Thinking..."):
                response = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "message": message,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
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
            st.session_state.total_messages = 0
            st.session_state.images_generated = 0
            st.success("🗑️ Conversation cleared!")
            st.rerun()
        else:
            st.error("❌ Failed to clear conversation on server")
    except Exception as e:
        st.error(f"❌ Error clearing conversation: {e}")

def display_chat_message(message_data, is_user=True):
    """Display a chat message with beautiful styling"""
    if is_user:
        # User message - right aligned
        st.markdown(f'''
        <div class="user-message message-animation">
            <div class="user-bubble">
                {message_data["content"]}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        # Assistant message - left aligned
        content = message_data["content"]
        language = message_data.get("language", "en")
        
        # Language mapping for display
        lang_display = {
            "en": "🇺🇸 English",
            "hi": "🇮🇳 हिंदी", 
            "es": "🇪🇸 Español",
            "fr": "🇫🇷 Français"
        }
        
        lang_badge = lang_display.get(language, f"🌍 {language.upper()}")
        
        st.markdown(f'''
        <div class="assistant-message message-animation">
            <div class="assistant-bubble">
                {content}
                <div class="language-badge">{lang_badge}</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Display image if present - with controlled size
        if "image" in message_data and message_data["image"]:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    message_data["image"], 
                    caption="🎨 Generated by EchoVision",
                    width=300,  # Fixed width for consistency
                    use_column_width=False
                )

def create_sidebar():
    """Create beautiful sidebar with dashboard"""
    with st.sidebar:
        # App branding
        st.markdown('''
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #667eea; font-size: 2rem; margin: 0;">🔮 EchoVision</h1>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">AI-Powered Chat & Image Generation</p>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # API Status
        st.markdown("### 🔌 Connection Status")
        api_healthy, api_data = check_api_status()
        st.session_state.api_status = "online" if api_healthy else "offline"
        
        if api_healthy:
            st.markdown('<div class="status-online">✅ Connected & Ready</div>', unsafe_allow_html=True)
            if api_data:
                col1, col2 = st.columns(2)
                with col1:
                    llm_status = "🧠" if api_data.get('mistral_loaded') else "❌"
                    st.metric("LLM", llm_status)
                with col2:
                    img_status = "🎨" if api_data.get('image_gen_loaded') else "❌"
                    st.metric("Image Gen", img_status)
                
                gpu_status = api_data.get('gpu_available', 'unknown').upper()
                st.info(f"💻 **Device:** {gpu_status}")
        else:
            st.markdown('<div class="status-offline">❌ Disconnected</div>', unsafe_allow_html=True)
            st.error("Please start the backend server first!")
            st.code("python main.py", language="bash")
        
        st.markdown("---")
        
        # Session Statistics
        if api_healthy:
            st.markdown("### 📊 Session Stats")
            stats = get_api_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Messages", st.session_state.total_messages)
            with col2:
                st.metric("Images", st.session_state.images_generated)
            
            if stats:
                st.metric("Active Sessions", stats.get("active_sessions", 0))
                st.metric("Total Conversations", stats.get("total_conversations", 0))
        
        st.markdown("---")
        
        # Language Support
        st.markdown("### 🌍 Language Support")
        languages = [
            ("🇺🇸", "English", "Full support"),
            ("🇮🇳", "हिंदी", "पूर्ण समर्थन"),
            ("🇪🇸", "Español", "Soporte completo"),
            ("🇫🇷", "Français", "Support complet")
        ]
        
        for flag, lang, desc in languages:
            st.markdown(f"**{flag} {lang}** - {desc}")
        
        st.markdown("---")
        
        # Image Generation Info
        st.markdown("### 🎨 Image Generation")
        st.markdown("""
        **Trigger phrases:**
        - "Create an image of..."
        - "Generate a picture of..."
        - "Draw me..."
        - "Show me an image..."
        - Or equivalent in any supported language
        """)
        
        st.markdown("---")
        
        # Session Management
        st.markdown("### 💬 Session Controls")
        st.markdown(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
        
        with col2:
            if st.button("🔄 New Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.total_messages = 0
                st.session_state.images_generated = 0
                st.success("New session started!")
                st.rerun()
        
        st.markdown("---")
        
        # About
        st.markdown("### ℹ️ About EchoVision")
        st.markdown("""
        EchoVision combines the power of **Mistral 7B** for intelligent conversations 
        with **Stable Diffusion** for creative image generation.
        
        **Features:**
        - 🤖 Advanced AI conversations
        - 🎨 High-quality image generation  
        - 🌍 Multi-language support
        - 💾 Conversation memory
        - ⚡ GPU-optimized performance
        """)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Create sidebar
    create_sidebar()
    
    # Main header
    st.markdown('''
    <div class="main-header">
        <h1>🔮 EchoVision</h1>
        <p>Your intelligent companion for conversations and creative visuals</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Check API status
    api_healthy, _ = check_api_status()
    
    if not api_healthy:
        st.error("⚠️ **Backend API is not running.** Please start the server first:")
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
    
    # Chat interface
    st.markdown("### 💬 Chat with EchoVision")
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message, is_user=message["role"] == "user")
    
    # Chat input
    if prompt := st.chat_input("Type your message in any supported language... ✨", key="chat_input"):
        # Add user message
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        st.session_state.total_messages += 1
        
        # Display user message immediately
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
                    st.session_state.images_generated += 1
                    
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
    st.markdown('''
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🚀 Powered by <strong>Mistral 7B</strong> + <strong>Stable Diffusion</strong> | 
        💻 Local AI Processing | 🌍 Multi-language Support</p>
        <p style="font-size: 0.8rem;">EchoVision - Where conversation meets creativity ✨</p>
    </div>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
