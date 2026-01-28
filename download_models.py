from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler
import os

import sys
sys.path.insert(0, '/app')

from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline

# Get HF token from environment
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN or HUGGING_FACE_HUB_TOKEN environment variable not set!")

print("Downloading FLUX 2 models...")
model_path = "black-forest-labs/FLUX.2-dev"

AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae", token=hf_token)
Flux2Transformer2DModel.from_pretrained(model_path, subfolder="transformer", token=hf_token)
FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler", token=hf_token)

print("Downloading Mistral text encoder...")
mistral_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

AutoProcessor.from_pretrained(mistral_path, token=hf_token)
Mistral3ForConditionalGeneration.from_pretrained(mistral_path, token=hf_token)

print("All models downloaded successfully!")
