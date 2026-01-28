import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze, qint8, quantize
import os

# Add /app to path FIRST
import sys
sys.path.insert(0, '/app')

from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline

# Get HF token from environment
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    print("WARNING: HF_TOKEN environment variable not set! Model downloads will fail for gated models.")

# Global pipeline (loaded once at startup)
PIPELINE = None
DEVICE = None

def load_models():
    """Load models once - called at container startup"""
    global PIPELINE, DEVICE
    
    if PIPELINE is not None:
        return PIPELINE, DEVICE
    
    print("Loading models...")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    model_path = "black-forest-labs/FLUX.2-dev"
    mistral_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    lora_path = "/app/models/pytorch_lora_weights.safetensors"
    
    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype, token=hf_token
    )
    vae.to(DEVICE)
    print(f"VAE loaded to {DEVICE}")
    
    # Load and quantize text encoder on CPU
    print("Loading text encoder...")
    processor = AutoProcessor.from_pretrained(mistral_path, token=hf_token)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        mistral_path, torch_dtype=dtype, low_cpu_mem_usage=True, token=hf_token
    )
    print("Quantizing text encoder...")
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)
    text_encoder.to(torch.device("cpu"), dtype=dtype)
    print("Text encoder loaded and quantized on CPU")
    
    # Load transformer
    print("Loading transformer...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, token=hf_token
    )
    transformer.to(DEVICE)
    print(f"Transformer loaded to {DEVICE}")
    
    # Load scheduler
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", token=hf_token
    )
    print("Scheduler loaded")
    
    # Create pipeline
    print("Creating pipeline...")
    pipeline = Flux2Pipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=processor,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.enable_model_cpu_offload()
    print("Pipeline created with CPU offloading enabled")
    
    # Load LoRA
    print(f"Loading LoRA from {lora_path}...")
    pipeline.load_lora_weights(lora_path)
    print("LoRA weights loaded")
    
    PIPELINE = pipeline
    print("✅ All models loaded successfully!")
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
print("Pre-loading models at startup...")
load_models()
