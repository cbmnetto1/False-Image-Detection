import tkinter as tk
from tkinter import filedialog, Label, Button
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'AI-vs-Human-Classifier.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match model input
    img = img.astype("float32") / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict image class
def predict_image():
    global img_label, result_label
    file_path = filedialog.askopenfilename()
    
    if not file_path:
        return
    
    # Display selected image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Process and predict
    processed_img = preprocess_image(file_path)
    prediction = model.predict(processed_img)[0][0]
    result = "AI-Generated" if prediction > 0.5 else "Human-Generated"

    # Update result label with styled output
    result_label.config(text=f"Result: {result}", fg="red" if result == "AI-Generated" else "green")

# Create the GUI window
root = tk.Tk()
root.title("AI vs Human Image Detector")
root.geometry("400x500")

# Title Label
title_label = Label(root, text="AI vs Human Image Detector", font=("Helvetica", 16, "bold"))
title_label.pack(pady=20)

# Image Label (Empty Initially)
img_label = Label(root)
img_label.pack(pady=10)

# Upload Button
upload_btn = Button(root, text="Upload Image", font=("Helvetica", 12), padx=10, pady=5, command=predict_image)
upload_btn.pack(pady=10)

# Prediction Result Label
result_label = Label(root, text="", font=("Helvetica", 14, "bold"))
result_label.pack(pady=20)

# Run GUI loop
root.mainloop()