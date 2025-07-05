from flask import Flask, redirect, request
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define captioning function
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# Create Gradio interface
gr_interface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="A Simple Image Caption Generator",
    description="Upload an image and the model will generate a caption."
)

# Launch Gradio inside Flask
app = Flask(__name__)

@app.route("/")
def index():
    return redirect("/gradio")

@app.route("/gradio")
def gradio_app():
    return gr_interface.launch(share=False, inline=True, prevent_thread_lock=True)

# Run the app
if __name__ == "__main__":
    app.run(port=5000)
