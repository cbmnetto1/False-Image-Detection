import os, random, shutil

origem = r"C:\Users\celso\Downloads\train_data"
destino = r"C:\DevPrograms\False Image\False-Image-Detection\dataset\train_data"

os.makedirs(destino, exist_ok=True)
arquivos = [os.path.join(origem, f) for f in os.listdir(origem) if os.path.isfile(os.path.join(origem, f))]

selecionados = random.sample(arquivos, 5000)

for arq in selecionados:
    shutil.move(arq, destino)
