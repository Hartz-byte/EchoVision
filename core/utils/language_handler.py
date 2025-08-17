"""
Multi-language support handler for English, Hindi, Spanish, and French
"""
import logging
from langdetect import detect, DetectorFactory
from typing import Dict, Optional
import re

# Set seed for consistent language detection
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class LanguageHandler:
    def __init__(self):
        """Initialize language handler with support for EN, HI, ES, FR"""
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi', 
            'es': 'Spanish',
            'fr': 'French'
        }
        
        # System prompts for each language
        self.system_prompts = {
            'en': """You are a helpful AI assistant. Respond naturally in English. 
If the user asks you to generate, create, make, draw, or show an image, respond with 'IMAGE_REQUEST:' followed by a detailed English description for image generation.

Example:
User: "Create an image of a sunset"
Assistant: "I'll create a beautiful sunset image for you.

IMAGE_REQUEST: A breathtaking sunset scene with vibrant orange and pink colors painting the sky, silhouetted mountains in the distance, and gentle clouds scattered across the horizon"

Always respond in English and be helpful and informative.""",

            'hi': """आप एक सहायक AI असिस्टेंट हैं। हिंदी में स्वाभाविक रूप से जवाब दें।
यदि उपयोगकर्ता आपसे कोई छवि बनाने, उत्पन्न करने, तैयार करने, या दिखाने को कहता है, तो 'IMAGE_REQUEST:' के साथ जवाब दें और फिर छवि निर्माण के लिए एक विस्तृत अंग्रेजी विवरण दें।

उदाहरण:
उपयोगकर्ता: "सूर्यास्त की एक छवि बनाएं"
असिस्टेंट: "मैं आपके लिए एक सुंदर सूर्यास्त की छवि बनाऊंगा।

IMAGE_REQUEST: A breathtaking sunset scene with vibrant orange and pink colors painting the sky, silhouetted mountains in the distance, and gentle clouds scattered across the horizon"

हमेशा हिंदी में जवाब दें और सहायक तथा जानकारीपूर्ण रहें।""",

            'es': """Eres un asistente de IA útil. Responde naturalmente en español.
Si el usuario te pide generar, crear, hacer, dibujar o mostrar una imagen, responde con 'IMAGE_REQUEST:' seguido de una descripción detallada en inglés para la generación de imágenes.

Ejemplo:
Usuario: "Crea una imagen de una puesta de sol"
Asistente: "Crearé una hermosa imagen de puesta de sol para ti.

IMAGE_REQUEST: A breathtaking sunset scene with vibrant orange and pink colors painting the sky, silhouetted mountains in the distance, and gentle clouds scattered across the horizon"

Siempre responde en español y sé útil e informativo.""",

            'fr': """Vous êtes un assistant IA utile. Répondez naturellement en français.
Si l'utilisateur vous demande de générer, créer, faire, dessiner ou montrer une image, répondez avec 'IMAGE_REQUEST:' suivi d'une description détaillée en anglais pour la génération d'images.

Exemple:
Utilisateur: "Crée une image d'un coucher de soleil"
Assistant: "Je vais créer une belle image de coucher de soleil pour vous.

IMAGE_REQUEST: A breathtaking sunset scene with vibrant orange and pink colors painting the sky, silhouetted mountains in the distance, and gentle clouds scattered across the horizon"

Répondez toujours en français et soyez utile et informatif."""
        }
        
        # Image request keywords for each language
        self.image_keywords = {
            'en': ['image', 'picture', 'photo', 'generate', 'create', 'draw', 'make', 'show me', 'paint', 'design'],
            'hi': ['छवि', 'तस्वीर', 'फोटो', 'बनाओ', 'दिखाओ', 'तैयार करो', 'चित्र', 'स्केच'],
            'es': ['imagen', 'foto', 'generar', 'crear', 'hacer', 'mostrar', 'dibujar', 'pintar', 'diseñar'],
            'fr': ['image', 'photo', 'générer', 'créer', 'faire', 'montrer', 'dessiner', 'peindre', 'concevoir']
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Language code (en, hi, es, fr) or 'en' as default
        """
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', '', text.strip())
            
            if len(cleaned_text) < 3:
                return 'en'  # Default to English for very short texts
            
            detected = detect(cleaned_text)
            
            # Map detected language to supported languages
            if detected in self.supported_languages:
                logger.debug(f"Detected language: {detected} ({self.supported_languages[detected]})")
                return detected
            
            # Handle common variations
            language_mapping = {
                'ca': 'es',  # Catalan -> Spanish
                'pt': 'es',  # Portuguese -> Spanish
                'it': 'fr',  # Italian -> French
                'de': 'en',  # German -> English
                'nl': 'en',  # Dutch -> English
            }
            
            if detected in language_mapping:
                mapped_lang = language_mapping[detected]
                logger.debug(f"Mapped {detected} to {mapped_lang}")
                return mapped_lang
            
            # Default to English for unsupported languages
            logger.debug(f"Language {detected} not supported, defaulting to English")
            return 'en'
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        return self.supported_languages.get(lang_code, 'English')
    
    def create_contextual_prompt(self, user_message: str, conversation_history: str, 
                               detected_lang: str) -> str:
        """
        Create a contextual prompt for the LLM
        
        Args:
            user_message: Current user message
            conversation_history: Previous conversation context
            detected_lang: Detected language code
            
        Returns:
            Formatted prompt for the LLM
        """
        system_prompt = self.system_prompts.get(detected_lang, self.system_prompts['en'])
        
        # Build the complete prompt
        prompt_parts = [
            "[INST]",
            system_prompt
        ]
        
        # Add conversation history if available
        if conversation_history and conversation_history.strip():
            prompt_parts.extend([
                "\n\nPrevious conversation context:",
                conversation_history[-1000:],  # Limit context to avoid token overflow
                ""
            ])
        
        # Add current user message
        prompt_parts.extend([
            f"Current user message: {user_message}",
            "\nAssistant:"
        ])
        
        prompt_parts.append("[/INST]")
        
        final_prompt = "\n".join(prompt_parts)
        logger.debug(f"Created prompt for language {detected_lang}, length: {len(final_prompt)}")
        
        return final_prompt
    
    def is_image_request(self, text: str, detected_lang: str = None) -> bool:
        """
        Check if the user is requesting image generation
        
        Args:
            text: User input text
            detected_lang: Optional detected language code
            
        Returns:
            True if image generation is requested
        """
        text_lower = text.lower()
        
        # If we have detected language, check its keywords first
        if detected_lang and detected_lang in self.image_keywords:
            keywords = self.image_keywords[detected_lang]
            for keyword in keywords:
                if keyword in text_lower:
                    return True
        
        # Check all language keywords as fallback
        for lang, keywords in self.image_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return True
        
        return False
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages"""
        return self.supported_languages.copy()
    
    def get_stats(self) -> dict:
        """Get language handler statistics"""
        return {
            "supported_languages": list(self.supported_languages.keys()),
            "language_names": list(self.supported_languages.values()),
            "total_languages": len(self.supported_languages),
            "image_keywords_total": sum(len(keywords) for keywords in self.image_keywords.values())
        }
