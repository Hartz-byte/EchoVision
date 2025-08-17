"""
Stable Diffusion Image Generation Handler
"""
import torch
import logging
from diffusers import StableDiffusionPipeline
from PIL import Image
import gc
import os

logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self, model_path: str):
        """
        Initialize Stable Diffusion pipeline
        
        Args:
            model_path: Path to local safetensors file or HuggingFace model name
        """
        self.model_path = model_path
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading Stable Diffusion model from {model_path}")
            logger.info(f"Device: {self.device}")
            
            # Check if it's a local file
            if os.path.isfile(model_path) and model_path.endswith(('.ckpt', '.safetensors')):
                logger.info("Loading from local safetensors file...")
                self.pipe = StableDiffusionPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=model_path.endswith('.safetensors'),
                    load_safety_checker=False
                )
            else:
                logger.info("Loading from HuggingFace repository...")
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                )
            
            if self.device == "cuda":
                # RTX 3050 4GB VRAM optimizations
                logger.info("Applying VRAM optimizations for RTX 3050...")
                
                # Most important: Sequential CPU offload
                self.pipe.enable_sequential_cpu_offload()
                
                # Enable attention slicing for memory efficiency
                self.pipe.enable_attention_slicing()
                
                # Enable VAE slicing to reduce VRAM usage
                if hasattr(self.pipe, 'enable_vae_slicing'):
                    self.pipe.enable_vae_slicing()
                
                # Try xformers for additional efficiency
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xformers memory efficient attention enabled")
                except Exception as e:
                    logger.info("⚠️ xformers not available, using standard attention")
                
                logger.info("✅ VRAM optimizations applied successfully")
            else:
                logger.warning("⚠️ CUDA not available, using CPU (will be slow)")
            
            logger.info("✅ Stable Diffusion model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading Stable Diffusion model: {e}")
            raise
    
    def generate_image(self, prompt: str, negative_prompt: str = "",
                      num_inference_steps: int = 20, guidance_scale: float = 7.5,
                      width: int = 512, height: int = 512) -> Image.Image:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            negative_prompt: Things to avoid in the image
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            width: Image width
            height: Image height
            
        Returns:
            Generated PIL Image
        """
        try:
            logger.info(f"🎨 Generating image for prompt: {prompt[:50]}...")
            
            # Default negative prompt for better quality
            if not negative_prompt:
                negative_prompt = "blurry, bad quality, distorted, deformed, low resolution, ugly, watermark, text"
            
            # Clear VRAM before generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            # Generate image
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                )
                
                image = result.images[0]
            
            # Clear VRAM after generation
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info("✅ Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"❌ Error generating image: {e}")
            # Return a placeholder image on error
            placeholder = Image.new('RGB', (width, height), color='lightgray')
            return placeholder
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("GPU memory cleaned up")
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "model_type": "Stable Diffusion v1.5",
            "torch_dtype": "float16" if self.device == "cuda" else "float32"
        }
