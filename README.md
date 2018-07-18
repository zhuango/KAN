# KAN

![Supported Python versions](https://img.shields.io/badge/python-3.6-blue.svg)
![Supported Pytorch versions](https://img.shields.io/badge/Pytorch-0.2-blue.svg)

**Knowledge-aware Attention Network for Protein-Protein Interaction Extraction** Zhou, H., Liu Z., Ning S. et al. Submitted to ***Bioinformatics***

An implementation of Knowledge-aware attention networks (KAN) for protein-protein extraction task.

This code has been written using Pytorch 0.2.

## Import data
We have put word embeddings, entity and relation embeddings learned from TransE[1] to the folder of data.

## Basic example
Go to the model path and run:
```console
❱❱❱ python3 main.py
```
In this setting, the default hyperparameters are used. Or run in specific settings:
```
❱❱❱ python3 main.py --trainPath ../data/train.txt --testPath ../data/test.txt --batchSize 100 --wd 100 --ed 100 --hop 2 --clas 2 --epoch 20 --wePath ../data/wordEmb/bio-word2id100 --w2IDPath ../data/wordEmb/bio-embed100 --eePath ../data/KB/entity2vec.vec --rePath ../data/KB/relation2vec.vec --t2idPath ../data/KB/triple2id.txt --e2idPath ../data/KB/entity2id.txt --paraPath ./parameters/ --results ./results/
```

the option you can choose are:
- `--trainPath` path of train dataset.
- `--testPath` path of test dataset.
- `--batchSize` batch size.
- `--wd` dimension of word embedding.
- `--ed` dimension of entity embedding learned from TransE.
- `--hop` number of hop.
- `--clas` number of class.
- `--epoch` number of iterations.
- `--wePath` path of word embedding file.
- `--w2IDPath` path of file that contains mapping from word to its number.
- `--eePath` path of entity embedding file.
- `--rePath` path of relation embedding file.
- `--t2idPath` path of file that contains the triples.
- `--e2idPath` path of file that contains mapping from Entrez Gene ID to number.
- `--paraPath` path of model parameters.
- `--results` path where the results write to.

# Reference

[1] Bordes, Antoine, et al. Translating embeddings for modeling multi-relational data. Proceedings of NIPS, 2013.