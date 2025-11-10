

# False Image Detection

Projeto desenvolvido para a disciplina de Processamento Digital de Imagem. O objetivo é treinar um classificador para distinguir imagens geradas por IA de imagens humanas.

Integrantes:
- Bruno Vicente
- Celso Bezerra
- Gregorio de Albuquerque
- Henrique Azevedo
- Henrique Rojas

## Descrição
Este projeto treina e avalia uma rede neural convolucional para classificação binária (AI vs Human). O script principal é o arquivo [script.ipynb](script.ipynb) e contém a configuração, carregamento de dados, definição do modelo, treino, avaliação e previsões.

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

