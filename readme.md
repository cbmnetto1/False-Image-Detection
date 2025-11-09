

# False Image Detection

Projeto desenvolvido para a disciplina de Processamento Digital de Imagem. O objetivo é treinar um classificador para distinguir imagens geradas por IA de imagens humanas.

Integrantes:
- Bruno Vicente
- Celso Bezerra
- Gregorio de Albuquerque
- Henrique Azevedo
- Henrique Rojas

## Descrição
Este projeto treina e avalia uma rede neural convolucional para classificação binária (AI vs Human). O script principal é o arquivo [script.py](script.py) e contém a configuração, carregamento de dados, definição do modelo, treino, avaliação e previsões.

## Estrutura esperada do dataset
Coloque o diretório `dataset/` no mesmo nível de `script.py`. O diretório deve conter:
- `train.csv` — colunas esperadas: `file_name`, `label`
- `test.csv` — colunas esperadas: `id`

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

Instalação:
```sh
pip install tensorflow pandas numpy matplotlib

