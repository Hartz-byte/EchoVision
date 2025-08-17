"""
Mistral LLM Handler for local model inference
"""
import logging
from llama_cpp import Llama
from typing import Optional
import os

logger = logging.getLogger(__name__)

class MistralHandler:
    def __init__(self, model_path: str, n_gpu_layers: int = 20, n_ctx: int = 4096, n_threads: int = 8):
        """
        Initialize Mistral model handler
        
        Args:
            model_path: Path to the GGUF model file
            n_gpu_layers: Number of layers to offload to GPU
            n_ctx: Context window size
            n_threads: Number of CPU threads to use
        """
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            logger.info(f"Loading Mistral model from {model_path}")
            logger.info(f"GPU layers: {n_gpu_layers}, Context: {n_ctx}, Threads: {n_threads}")
            
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                verbose=False,
                use_mmap=True,
                use_mlock=False,
                n_batch=512,
                f16_kv=True  # Use float16 for key-value cache to save VRAM
            )
            logger.info("Mistral model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Mistral model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, 
                         top_p: float = 0.9, stop: Optional[list] = None) -> str:
        """
        Generate response from the model
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated response text
        """
        try:
            if stop is None:
                stop = ["</s>", "[INST]", "[/INST]", "\n\nUser:", "\n\nHuman:"]
            
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=stop,
                repeat_penalty=1.1
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean up response
            if not generated_text:
                return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def create_instruct_prompt(self, message: str, system_prompt: str = "") -> str:
        """
        Create properly formatted instruct prompt for Mistral
        
        Args:
            message: User message
            system_prompt: System prompt (optional)
            
        Returns:
            Formatted prompt
        """
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{message} [/INST]"
        else:
            return f"[INST] {message} [/INST]"
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "n_gpu_layers": self.n_gpu_layers,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "model_type": "Mistral 7B Instruct v0.2"
        }
