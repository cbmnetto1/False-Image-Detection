# 🧠 False Image Detection  

Projeto desenvolvido para a disciplina de **Processamento Digital de Imagens (PDI)**.  
O objetivo é **treinar um classificador capaz de distinguir imagens geradas por Inteligência Artificial (IA) de imagens reais produzidas por humanos**.  

---

## 👥 Integrantes  
- Bruno Vicente  
- Celso Bezerra  
- Gregorio de Albuquerque  
- Henrique Azevedo  
- Henrique Rojas  

---

## 📝 Descrição do Projeto  
O projeto implementa uma **Rede Neural Convolucional (CNN)** para realizar **classificação binária** (IA vs. Humano).  

O script principal é o arquivo [`script.ipynb`](script.ipynb), que contém:  
1. **Configuração do ambiente**  
2. **Carregamento e pré-processamento dos dados**  
3. **Definição da arquitetura da rede**  
4. **Treinamento do modelo**  
5. **Avaliação de desempenho**  
6. **Geração de previsões** 
## Dataset
O modelo já foi treinado. Mas caso queira treina-lo novamente:
Crie o diretório `dataset/` no mesmo nível de `script.ipynb`. O diretório deve conter:
- `train.csv` 
- `test.csv` 
- `train_data` - Contem as imagens de treino
- `test_data_v2` - Contem as imagens de teste

As imagens referenciadas nas CSVs devem estar dentro de `dataset/`.

## Referência do Dataset
O conjunto de dados utilizado foi retirado do Kaggle:
[AI vs Human Generated Dataset](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset/data)

## Requisitos
- Python 3.10
- tensorflow
- pandas
- numpy
- matplotlib
- keras
- pillow
- opencv-python

Instalação:
```sh
pip install -r requirements.txt

