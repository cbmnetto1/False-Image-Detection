import tkinter as tk
from tkinter import filedialog, Label, Button, ttk, messagebox
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
import os
import datetime

# Caminho do modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'AI-vs-Human-Classifier.keras')
model = tf.keras.models.load_model(MODEL_PATH)

# Função para pré-processar imagem
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Redimensiona para o input do modelo
    img = img.astype("float32") / 255.0  # Normaliza
    img = np.expand_dims(img, axis=0)  # Adiciona dimensão batch
    return img

# Função para prever a classe da imagem
def predict_image():
    global img_label, result_label

    file_path = filedialog.askopenfilename(
        title="Selecione uma imagem",
        filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
    )
    
    if not file_path:
        return

    # Mostra a imagem selecionada
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Mostra mensagem de status e barra de progresso
    status_label.config(text="Analisando imagem...", fg="blue")
    progress_bar.start()

    # Executa previsão após pequena pausa para atualizar GUI
    root.after(100, lambda: make_prediction(file_path))

def make_prediction(file_path):
    try:
        processed_img = preprocess_image(file_path)
        prediction = model.predict(processed_img)[0][0]
        result = "Gerado por IA" if prediction > 0.5 else "Gerado por Humano"

        # Atualiza label de resultado
        result_label.config(
            text=f"Resultado: {result}",
            fg="red" if result == "Gerado por IA" else "green"
        )

        # Atualiza status e para barra
        status_label.config(text="Análise concluída!", fg="black")
        progress_bar.stop()

        # Adiciona ao histórico
        add_to_history(file_path, result)

    except Exception as e:
        progress_bar.stop()
        status_label.config(text="Erro ao processar imagem.", fg="red")
        messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

# Adiciona resultado ao histórico
def add_to_history(image_path, result):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    filename = os.path.basename(image_path)
    history_list.insert(0, f"[{timestamp}] {filename} → {result}")

# Cria a janela principal
root = tk.Tk()
root.title("Detector de Imagem Humano vs IA")
root.geometry("500x650")
root.resizable(False, False)

# Título
title_label = Label(root, text="Detector de Imagem Humano vs IA", font=("Helvetica", 16, "bold"))
title_label.pack(pady=15)

# Imagem
img_label = Label(root)
img_label.pack(pady=10)

# Botão de upload
upload_btn = Button(root, text="Selecionar Imagem", font=("Helvetica", 12), padx=10, pady=5, command=predict_image)
upload_btn.pack(pady=10)

# Barra de progresso
progress_bar = ttk.Progressbar(root, mode='indeterminate', length=250)
progress_bar.pack(pady=10)

# Label de status
status_label = Label(root, text="", font=("Helvetica", 10))
status_label.pack()

# Resultado
result_label = Label(root, text="", font=("Helvetica", 14, "bold"))
result_label.pack(pady=20)

# Histórico
history_label = Label(root, text="Histórico de análises", font=("Helvetica", 12, "bold"))
history_label.pack(pady=5)

history_list = tk.Listbox(root, height=8, width=60)
history_list.pack(pady=5)

# Inicia o loop
root.mainloop()
