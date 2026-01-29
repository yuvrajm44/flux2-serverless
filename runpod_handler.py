import runpod
import os
import io
import gc
from PIL import Image
import requests
from inference import generate_image

def download_image(url):
    """Download image from URL"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def handler(event):
    """RunPod serverless handler"""
    try:
        input_data = event["input"]
        request_id = event.get("id", "unknown")
        
        # Get inputs
        prompt = input_data["prompt"]
        reference_urls = input_data["reference_images"]
        num_steps = input_data.get("num_steps", 20)
        guidance_scale = input_data.get("guidance_scale", 3.5)
        width = input_data.get("width", 1024)
        height = input_data.get("height", 1024)
        seed = input_data.get("seed", 42)
        
        # Download reference images
        reference_images = [download_image(url) for url in reference_urls]
        
        # Generate image
        output_image = generate_image(
            prompt=prompt,
            reference_images=reference_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed
        )
        
        # FREE reference images immediately
        del reference_images
        gc.collect()
        
        # Save to network volume instead of base64
        output_dir = "/runpod-volume/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = f"{output_dir}/{request_id}.jpg"
        output_image.save(output_path, format="JPEG", quality=95)
        
        # FREE output image immediately after saving
        del output_image
        gc.collect()
        
        return {
            "image_path": output_path,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

runpod.serverless.start({"handler": handler})
