

# KAN

![Supported Python versions](https://img.shields.io/badge/python-3.6-blue.svg)
![Supported Pytorch versions](https://img.shields.io/badge/pytorch-0.4-blue.svg)

**Knowledge-aware Attention Network for Protein-Protein Interaction Extraction** Zhou, H., Liu Z., Ning S. et al. Accepted by ***The Journal of Biomedical informatics***

Una implementación de redes de atención con conocimiento (KAN) para la tarea de extracción de interacciones proteína-proteína.

Este código ha sido escrito utilizando Pytorch 0.4.

## Importar datos
Hemos colocado las incrustaciones de palabras, así como las incrustaciones de entidades y relaciones aprendidas de TransE[1] en la carpeta de datos.

## Ejemplo básico
Ve a la ruta del modelo y ejecuta:
```console
❱❱❱ python3 main.py
```
En esta configuración, se utilizan los hiperparámetros predeterminados. O ejecuta con configuraciones específicas:
```
❱❱❱ python3 main.py --trainPath ../data/train.txt --validPath ../data/valid.txt --testPath ../data/test.txt --trainGold ../data/trainGold.txt --testGold ../data/testGold.txt --batchSize 100 --wd 100 --ed 100 --hop 2 --clas 2 --epoch 20 --wePath ../data/wordEmb/bio-word2id100 --w2IDPath ../data/wordEmb/bio-embed100 --eePath ../data/KB/entity2vec.vec --rePath ../data/KB/relation2vec.vec --t2idPath ../data/KB/triple2id.txt --e2idPath ../data/KB/entity2id.txt --paraPath ./parameters/kan --results ./results/ --training True --lr 0.1 --wdecay 0.0 --validsetR 0.15
```

Las opciones que puedes elegir son:
- `--trainPath` ruta del conjunto de datos de entrenamiento.
- `--validPath` ruta del conjunto de datos de validación.
- `--testPath` ruta del conjunto de datos de prueba.
- `--trainGold` ruta de las triples de referencia del conjunto de datos de entrenamiento 
- `--testGold` ruta de las triples de referencia del conjunto de datos de prueba 
- `--batchSize` tamaño del lote.
- `--wd` dimensión de la incrustación de palabras.
- `--ed` dimensión de la incrustación de entidades aprendida de TransE.
- `--hop` número de saltos.
- `--clas` número de clases.
- `--epoch` número de iteraciones.
- `--wePath` ruta del archivo de incrustaciones de palabras.
- `--w2IDPath` ruta del archivo que contiene el mapeo de palabras a sus números.
- `--eePath` ruta del archivo de incrustaciones de entidades.
- `--rePath` ruta del archivo de incrustaciones de relaciones.
- `--t2idPath` ruta del archivo que contiene las triples.
- `--e2idPath` ruta del archivo que contiene el mapeo del ID de Entrez Gene a un número.
- `--paraPath` ruta de los parámetros del modelo.
- `--results` ruta donde se guardan los resultados.
- `--training` valor booleano para indicar si se debe entrenar o no. Una cadena no vacía indica la fase de entrenamiento. Una cadena vacía indica la fase de prueba.
- `--lr` tasa de aprendizaje.
- `--wdecay` decaimiento de peso.
- `--validsetR` este parámetro ([0.0, 1.0]) indica qué proporción de documentos del conjunto de entrenamiento se seleccionará para el conjunto de validación.

# Referencias

[1] Bordes, Antoine, et al. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.
