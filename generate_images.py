#!/usr/bin/env python3
import os
import argparse
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import glob

def get_image_prompts():
    """Get all image prompts from txt files in the images directory."""
    prompts = {}
    for txt_file in glob.glob('images/*.txt'):
        img_name = os.path.splitext(os.path.basename(txt_file))[0]
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            prompts[img_name] = content
    return prompts

def generate_image(prompt, api_token, dry_run=False):
    """Generate an image using the Hugging Face API."""
    if dry_run:
        print(f"Would generate image for prompt:\n{prompt}\n")
        return None

    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    response = requests.post(API_URL, headers=headers, json={
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "watermark, text, poor quality, distorted, blurry",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        }
    })
    
    if response.ok:
        return Image.open(BytesIO(response.content))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def save_image(image, filename):
    """Save the generated image."""
    image.save(filename)
    print(f"Saved image to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate images from text prompts")
    parser.add_argument("--api-token", required=True, help="Hugging Face API token")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without generating images")
    args = parser.parse_args()

    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    # Get prompts from txt files
    prompts = get_image_prompts()
    
    # Generate images for each prompt
    for img_name, prompt in tqdm(prompts.items()):
        output_path = f"images/{img_name}.jpg"
        
        print(f"\nGenerating {img_name}...")
        image = generate_image(prompt, args.api_token, args.dry_run)
        
        if image and not args.dry_run:
            save_image(image, output_path)

if __name__ == "__main__":
    main()