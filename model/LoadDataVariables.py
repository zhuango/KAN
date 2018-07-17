#!/usr/bin/python3
import torch
from torch.autograd import Variable
from collections import defaultdict, OrderedDict
import numpy as np

def LoadWordVectors(wordVectorFile, dimension):
    #word2vector = defaultdict( lambda:Variable(torch.FloatTensor(np.array(np.random.uniform(-0.01, 0.01, (dimension, 1)), dtype=np.float)), requires_grad=True) )
    word2vector = {}
    with open(wordVectorFile, 'r') as wordVectorsStream:
        for line in wordVectorsStream:
            items = line.strip().split(" ")
            if len(items) < dimension:
                continue
            word2vector[items[0]] = Variable(torch.FloatTensor(np.array([float(elem) for elem in items[1:]], dtype=np.float).reshape(dimension, 1)), requires_grad=True)
    return word2vector

def LoadWord2id(wordIdFile):
    word2id = defaultdict(lambda:0)
    with open(wordIdFile, 'r') as f:
        for line in f:
            items = line.strip().split(" ")
            if len(items) < 2:
                continue
            word2id[items[0]] = int(items[1])
    return word2id

def LoadEntityVectors(entityVectorFile, dimension):    
    entity2vector = OrderedDict()#defaultdict( lambda:Variable(torch.FloatTensor(np.array(np.random.uniform(-0.01, 0.01, (dimension, 1)), dtype=np.float)), requires_grad=True) )
    entity2vector['reserved'] = Variable(torch.FloatTensor(np.array(np.random.uniform(-0.0, 0.0, (dimension, 1)), dtype=np.float)), requires_grad=True)
    id = 0
    with open(entityVectorFile, 'r') as entityVectorsStream:
        for line in entityVectorsStream:
            items = line.strip().split("\t")
            if len(items) < dimension:
                continue
            entity2vector[str(id)] = Variable(torch.FloatTensor(np.array([float(elem) for elem in items], dtype=np.float).reshape(dimension, 1)), requires_grad=True)
            id += 1
    return entity2vector

def LoadSamples(corpusPath):
    samples = []
    with open(corpusPath, 'r') as corpusStream:
        for line in corpusStream:
            items = line.strip().split("\t")
            if len(items) < 2:
                continue
            samples.append(items)
    return samples

def LoadMulInstanceSamples(corpusPath):
    samples = []
    with open(corpusPath, 'r') as corpusStream:
        for line in corpusStream:
            sentenceCount = int(line.strip().split("\t")[0])
            sentences = []
            for i in range(sentenceCount):
                sentences.append(corpusStream.readline().strip().split("\t"))
            samples.append(sentences)
    return samples

def LoadEntity2Id(filename):
    mapping = defaultdict( lambda: str(len(mapping)) )
    with open(filename, 'r') as f:
        # skep the first line, which is a total number.
        f.readline()
        for line in f:
            items = line.strip().split("\t")
            mapping[items[0]] = items[1]
    return mapping

def LoadRelationVectors(filename, dimension, cuda=False, requires_grad=False):
    relation2vector = defaultdict(lambda: unrelated)
    if cuda:
        unrelated = Variable(torch.FloatTensor(np.array(np.random.uniform(-0.0, 0.0, (dimension, 1)), dtype=np.float)).cuda(),  requires_grad=requires_grad)
    else:
        unrelated = Variable(torch.FloatTensor(np.array(np.random.uniform(-0.0, 0.0, (dimension, 1)), dtype=np.float)),  requires_grad=requires_grad)

    id = 0
    #obsoleteInteraction = ["4", "", "", "", "", ""]
    with open(filename, 'r') as relationVectorsStream:
        for line in relationVectorsStream:
            items = line.strip().split("\t")
            if len(items) < dimension:
                continue
            # if id == 4:
            #     id += 1
            #     continue
            if cuda:
                relation2vector[str(id)] = Variable(torch.FloatTensor(np.array([float(elem) for elem in items], dtype=np.float).reshape(dimension, 1)).cuda(), requires_grad=True)
            else:
                relation2vector[str(id)] = Variable(torch.FloatTensor(np.array([float(elem) for elem in items], dtype=np.float).reshape(dimension, 1)), requires_grad=True)
            id += 1
    relation2vector['unknow'] = unrelated
    return relation2vector

def LoadTriples(filename):
    triple = {}
    with open(filename, 'r') as f:
        f.readline()
        for line in f:
            items = line.strip().split("\t")
            triple[items[0] + "_" + items[1]] = items[2]
            #triple[items[1] + "_" + items[0]] = items[2]
    return triple

def LoadEntityName2FormatID(filename):
    mapping = defaultdict(lambda key:key)
    with open(filename, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            mapping[items[1].lower()] = items[0]
    return mapping

if __name__ == "__main__":
    # word2vector = LoadWordVectors("/home/laboratory/corpus/en_vectorTable/en_vectors_50.txt", 50)
    # print(word2vector["<user>"])

    # entity2vector = LoadEntityVector("/home/laboratory/lab/BioCreative/codePlayer/Fast-TransX/transE/output/50d/entity2vec.vec", 50)
    # print(entity2vector["0"])
    
    # samples = LoadSamples("/home/laboratory/lab/BioCreative/codePlayer/preprocessing/corpus.txt")
    # print(samples[0])

    # entity2id = LoadEntity2Id("/home/laboratory/lab/BioCreative/codePlayer/Fast-TransX/data/entity2id.txt")
    # print(entity2id["Q8IMK5"])
    
    # triples = LoadTriples("/home/laboratory/lab/BioCreative/codePlayer/Fast-TransX/data/triple2id.txt")
    # print(triples)

    # mapping = LoadEntityName2FormatID("/home/laboratory/lab/sem2013/DDI/durg_dic/drugbank/eName2id.txt")
    # print(mapping)

    samples = LoadMulInstanceSamples("/home/laboratory/lab/BioCreative/2017/BC6/corpus_train_sentence_mulInstance.txt")
    print(samples[0])