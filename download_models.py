from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from diffusers import FlowMatchEulerDiscreteScheduler

import sys
sys.path.insert(0, '/app')

from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline

print("Downloading FLUX 2 models...")
model_path = "black-forest-labs/FLUX.2-dev"

AutoencoderKLFlux2.from_pretrained(model_path, subfolder="vae")
Flux2Transformer2DModel.from_pretrained(model_path, subfolder="transformer")
FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

print("Downloading Mistral text encoder...")
mistral_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

AutoProcessor.from_pretrained(mistral_path)
Mistral3ForConditionalGeneration.from_pretrained(mistral_path)

print("All models downloaded successfully!")