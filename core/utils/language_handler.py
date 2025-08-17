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

        CRITICAL RULE: NEVER include 'IMAGE_REQUEST:' in your response unless the user uses explicit image creation words like 'create image', 'generate picture', 'draw', 'make image', or 'show me image'.

        For normal conversations, greetings, questions, or general chat - respond normally WITHOUT any IMAGE_REQUEST.

        ONLY if user explicitly says something like:
        - "Create an image of..."
        - "Generate a picture of..."  
        - "Draw me..."
        - "Make an image of..."

        Then respond: "I'll create that image for you. IMAGE_REQUEST: [detailed description]"

        For everything else, just have a normal conversation. Do NOT add IMAGE_REQUEST to regular responses.

        Always respond in English and be helpful and informative.""",

            'hi': """आप एक सहायक AI असिस्टेंट हैं। हिंदी में स्वाभाविक रूप से जवाब दें।

        महत्वपूर्ण नियम: जब तक उपयोगकर्ता स्पष्ट रूप से 'छवि बनाएं', 'तस्वीर बनाओ', 'चित्र दिखाएं' जैसे शब्द न कहे, तब तक कभी भी 'IMAGE_REQUEST:' का उपयोग न करें।

        सामान्य बातचीत, अभिवादन, प्रश्न के लिए - सामान्य जवाब दें IMAGE_REQUEST के बिना।

        केवल तभी IMAGE_REQUEST का उपयोग करें जब उपयोगकर्ता स्पष्ट रूप से कहे:
        - "एक छवि बनाएं..."
        - "तस्वीर बनाओ..."  
        - "चित्र दिखाएं..."

        अन्यथा सामान्य बातचीत करें। नियमित उत्तरों में IMAGE_REQUEST न जोड़ें।

        हमेशा हिंदी में जवाब दें।""",

            'es': """Eres un asistente de IA útil. Responde naturalmente en español.

        REGLA CRÍTICA: NUNCA incluyas 'IMAGE_REQUEST:' en tu respuesta a menos que el usuario use palabras explícitas de creación de imágenes como 'crear imagen', 'generar foto', 'dibujar', 'hacer imagen'.

        Para conversaciones normales, saludos, preguntas - responde normalmente SIN ningún IMAGE_REQUEST.

        SOLO si el usuario dice explícitamente:
        - "Crea una imagen de..."
        - "Genera una foto de..."
        - "Dibuja..."
        - "Haz una imagen de..."

        Entonces responde: "Crearé esa imagen para ti. IMAGE_REQUEST: [descripción detallada]"

        Para todo lo demás, ten una conversación normal. NO agregues IMAGE_REQUEST a respuestas regulares.

        Siempre responde en español.""",

            'fr': """Vous êtes un assistant IA utile. Répondez naturellement en français.

        RÈGLE CRITIQUE: N'incluez JAMAIS 'IMAGE_REQUEST:' dans votre réponse sauf si l'utilisateur utilise des mots explicites de création d'images comme 'créer image', 'générer photo', 'dessiner', 'faire image'.

        Pour les conversations normales, salutations, questions - répondez normalement SANS aucun IMAGE_REQUEST.

        SEULEMENT si l'utilisateur dit explicitement:
        - "Crée une image de..."
        - "Génère une photo de..."
        - "Dessine..."
        - "Fais une image de..."

        Alors répondez: "Je vais créer cette image pour vous. IMAGE_REQUEST: [description détaillée]"

        Pour tout le reste, ayez une conversation normale. N'ajoutez PAS IMAGE_REQUEST aux réponses régulières.

        Répondez toujours en français."""
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
