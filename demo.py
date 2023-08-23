import torch
from transformers import CLIPProcessor
from diffusers import TextToImageDiffuser
from invisible_watermark import InvisibleWatermark
from torchvision.utils import save_image

def main():
    # Load the pre-trained CLIP model and the text-to-image diffuser
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    diffuser = TextToImageDiffuser.from_pretrained("CompVis/stable-diffusion")

    # Input text prompt
    text = "A beautiful sunset over the ocean"

    # Encode the text into a latent code using the CLIP model
    inputs = processor(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        latent = diffuser.encode_text(inputs.input_ids.to(device))

    # Decode the latent code into an image
    image = diffuser.generate_image(latent)

    # Apply an invisible watermark (optional)
    watermark = InvisibleWatermark()
    watermarked_image = watermark.embed(image, "Sample watermark")

    # Save the generated image
    save_image(watermarked_image, "output.png")

if __name__ == "__main__":
    main()