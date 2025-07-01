'''
Note: This code is adapted from Coursera's IBM AI Developer course. All credit goes to Sina Nazeri

'''

import gradio as gr #Gradio is a Python library used to rapidly create and share user-friendly web interfaces for machine learning models, APIs, and other Python functions
import numpy as np # NumPy (Numerical Python) is a fundamental open-source library in Python, widely used for scientific computing and data analysis
from PIL import Image #Python Imaging Library
from transformers import AutoProcessor, BlipForConditionalGeneration
#BLIP (Bootstrapped Language-Image Pretraining) is a vision-language pretraining (VLP) framework designed for both understanding and generation tasks

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base") # https://huggingface.co/Salesforce/blip-image-captioning-base
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')

    # Process the image
    inputs = processor(raw_image, return_tensors="pt")

    # Generate a caption for the image
    out = model.generate(**inputs,max_length=50)

    # Decode the generated tokens to text
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="PIXCRIBE",
    description="Generate Image Captions using Salesforce/blip-image-captioning-base from Huggingface."
)

iface.launch()
