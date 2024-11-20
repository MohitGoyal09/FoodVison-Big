import os
import gradio as gr
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names_path = os.path.join(os.path.dirname(__file__), "class_names.txt")
with open(class_names_path, "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

# Create model
effnetb2, effnetb2_transforms = create_effnetb2_model(
    num_classes=101,  # could also use len(class_names)
)

# Load saved weights
model_path = os.path.join(os.path.dirname(__file__), "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth")
effnetb2.load_state_dict(torch.load(f=model_path, map_location=torch.device("cpu")))

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    start_time = timer()
    img = effnetb2_transforms(img).unsqueeze(0)
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

# Title and description
title = "FoodVision Big 🍔👁"
description = "An EfficientNetB2 feature extractor model to classify food images into 101 different classes."


# Create examples list
example_dir = os.path.join(os.path.dirname(__file__), "examples")
example_list = [[os.path.join(example_dir, example)] for example in os.listdir(example_dir) if example.endswith(("png", "jpg", "jpeg"))]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    
)

demo.launch()
