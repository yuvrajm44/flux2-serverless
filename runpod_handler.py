import runpod
import base64
import io
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
        import gc
        
        input_data = event["input"]
        
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
        
        # FREE reference images immediately - don't need them anymore
        del reference_images
        gc.collect()
        
        # Convert to JPEG (smaller than PNG)
        buffered = io.BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        buffered.seek(0)
        img_base64 = base64.b64encode(buffered.read()).decode()
        
        # FREE output image and buffer
        del buffered
        del output_image
        gc.collect()
        
        return {
            "image": f"data:image/jpeg;base64,{img_base64}",
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}
