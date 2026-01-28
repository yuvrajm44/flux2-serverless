import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze, qint8, quantize

# Add /app to path FIRST
import sys
sys.path.insert(0, '/app')

# THEN import SimpleTuner modules
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline
from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2

# Global pipeline (loaded once at startup)
PIPELINE = None
DEVICE = None

def load_models():
    """Load models once - called at container startup"""
    global PIPELINE, DEVICE
    
    if PIPELINE is not None:
        return PIPELINE, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    model_path = "black-forest-labs/FLUX.2-dev"
    mistral_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    lora_path = "/app/models/pytorch_lora_weights.safetensors"
    
    # Load VAE
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype
    )
    vae.to(DEVICE)
    
    # Load and quantize text encoder on CPU
    processor = AutoProcessor.from_pretrained(mistral_path)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        mistral_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)
    text_encoder.to(torch.device("cpu"), dtype=dtype)
    
    # Load transformer
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype
    )
    transformer.to(DEVICE)
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler"
    )
    
    # Create pipeline
    pipeline = Flux2Pipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=processor,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.enable_model_cpu_offload()
    
    # Load LoRA
    pipeline.load_lora_weights(lora_path)
    
    PIPELINE = pipeline
    return PIPELINE, DEVICE

def generate_image(prompt, reference_images, num_steps, guidance_scale, width, height, seed):
    """Generate image using loaded pipeline"""
    pipeline, device = load_models()
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    output = pipeline(
        prompt=prompt,
        image=reference_images,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    return output.images[0]

# Pre-load models at import time
load_models()