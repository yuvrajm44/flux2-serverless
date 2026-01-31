import os
import sys
import psutil
import gc
# SET THESE BEFORE ANY OTHER IMPORTS
os.environ['HF_HOME'] = '/runpod-volume'
os.environ['TRANSFORMERS_CACHE'] = '/runpod-volume'
os.environ['HF_HUB_CACHE'] = '/runpod-volume'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600' 

# Add /app to path
sys.path.insert(0, '/app')

# CHECK NETWORK VOLUME MOUNT
if os.path.exists('/runpod-volume'):
    print("✅ Network volume detected at /runpod-volume")
    try:
        test_file = '/runpod-volume/.test_write'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("✅ Network volume is writable")
    except Exception as e:
        print(f"❌ Network volume exists but not writable: {e}")
else:
    print("❌ WARNING: Network volume NOT mounted")

# NOW import everything else
import torch
from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler
from optimum.quanto import freeze, qint8, quantize
from huggingface_hub import hf_hub_download

from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline


print("DEBUG - Environment variables:")
print(f"HF_TOKEN: {os.environ.get('HF_TOKEN', 'NOT SET')}")
print(f"HUGGING_FACE_HUB_TOKEN: {os.environ.get('HUGGING_FACE_HUB_TOKEN', 'NOT SET')}")
print(f"All env vars: {list(os.environ.keys())}")

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
    
    # Check if models already cached in network volume
    flux_cached = os.path.exists("/runpod-volume/models--black-forest-labs--FLUX.2-dev")
    mistral_cached = os.path.exists("/runpod-volume/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503")
    
    if flux_cached and mistral_cached:
        print("✅ Models found in network volume - using cached versions")
    else:
        print("📥 Models not found in network volume - downloading (first run only)")
    
    print("Downloading LoRA from HuggingFace...")
    lora_path = hf_hub_download(
        repo_id="michealscott/flux2-3030",
        filename="pytorch_lora_weights.safetensors",
        token=hf_token
    )
    print(f"LoRA downloaded to: {lora_path}")
    
    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(
        model_path, subfolder="vae", torch_dtype=dtype, token=hf_token,
        device_map="cuda"
    )
    print(f"VAE loaded to {DEVICE}")
    print(f"💾 RAM after VAE: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
    
    # Load and quantize text encoder on CPU
    print("Loading text encoder...")
    processor = AutoProcessor.from_pretrained(mistral_path, token=hf_token)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        mistral_path, torch_dtype=dtype, low_cpu_mem_usage=True, token=hf_token
    )
    print(f"💾 RAM after text encoder load: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
    print("Quantizing text encoder...")
    quantize(text_encoder, weights=qint8)
    freeze(text_encoder)
    print("✅ Text encoder loaded and quantized on CPU (waiting to move to GPU)")
    print(f"💾 RAM after quantization: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")

  
    gc.collect()
    torch.cuda.empty_cache()
    print(f"💾 RAM after cleanup: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
        
    # Load transformer
    print("Loading transformer...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=dtype, token=hf_token,
        device_map="cuda"
    )
    print("Quantizing transformer to int8 on GPU...")
    quantize(transformer, weights=qint8)
    freeze(transformer)
    print("✅ Transformer quantized to int8 on GPU")
    
    # ⭐ ADD THIS SAFETY CHECK ⭐
    gc.collect()
    torch.cuda.empty_cache()
    current_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 VRAM after transformer quantization: {current_vram:.2f} GB")

    # 3. NOW move Text Encoder to GPU (only if transformer is safely quantized)
    print("Moving text encoder to GPU...")
    text_encoder.to(torch.device("cuda"), dtype=dtype)
    print("✅ Text encoder moved to GPU")

    # Final VRAM check
    final_vram = torch.cuda.memory_allocated() / 1024**3
    print(f"💾 Final VRAM usage: {final_vram:.2f} GB")

    gc.collect()
    torch.cuda.empty_cache()

    print(f"Transformer loaded to {DEVICE}")
    print(f"💾 RAM after transformer: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
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
    
    # pipeline.enable_model_cpu_offload()  
    print("Pipeline created (CPU offloading disabled)")
    
    # ADD THIS:
    print(f"Text encoder device: {text_encoder.device}")
    print(f"Transformer device: {transformer.device}")
    print(f"VAE device: {vae.device}")
    

    print(f"RAM usage: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
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

    print(f"💾 RAM before inference: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Let pipeline handle generator creation based on model devices
    output = pipeline(
        prompt=prompt,
        image=reference_images,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    print(f"💾 RAM after inference: {psutil.Process().memory_info().rss / 1024**3:.2f} GB")
    # Get the result
    result_image = output.images[0]
    
    # Free pipeline output immediately
    del output
    gc.collect()
    torch.cuda.empty_cache()
    
    return result_image

# Pre-load models at import time
print("Pre-loading models at startup...")
load_models()
