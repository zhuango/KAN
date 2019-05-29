#/usr/bin/python3

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import argparse
import re

def extractAnnotationNode(node):
    # If there is no UniProt id
    # Then continue.
    notValidNode = False
    uniprotIdx = -1
    geneIdx = -1
    for i in range(len(node)):
        info = node[i]
        if "key" in info.attrib:
            key = info.attrib["key"]
        else:
            continue
        if key.upper() == "NCBI GENE":
            geneIdx = i

    uniprotIds = []
    if geneIdx != -1:
        idText = node[geneIdx].text
        uniprotIds = idText.split(",")
        if ";" in idText:
            uniprotIds = idText.split(";")
    else:
        notValidNode = True
    
    locationNode = node.find('location')
    # offset of crrent entity's tail char.
    offset = int(locationNode.attrib["offset"])
    length = int(locationNode.attrib["length"])

    return uniprotIds, offset, length, notValidNode
def dup(entityNodes):    
    entityCount = len(entityNodes)
    if entityCount < 1:
        return []
    # rough
    dupEntityNodes = [entityNodes[0]]
    for i in range(1, entityCount):
        if int(entityNodes[i].find('location').attrib["offset"]) == int(entityNodes[i-1].find('location').attrib["offset"]):
            continue
        else:
            dupEntityNodes.append(entityNodes[i])
    return dupEntityNodes
def writePassage(fulltextElement, labelPairs, pmid, formatCorpusStream):
    text = fulltextElement[2].text
    fulltextOffset = int(fulltextElement[1].text)

    sentences = [sent + ". " for sent in text.split(". ")]
    sentenceCount = len(sentences)
    sentenceOffsets = [fulltextOffset]
    for j in range(1, sentenceCount):
        sentenceOffsets.append(sentenceOffsets[j-1] + len(sentences[j-1]))

    childCount = len(fulltextElement)
    entityNodes = fulltextElement[3:childCount]
    # sort entity nodes by their offset.
    entityNodes = sorted(entityNodes, key=lambda node: int(node.find('location').attrib["offset"]))
    # rough
    entityNodes = dup(entityNodes)
    entityCount = len(entityNodes)
    # the entity node starts with index of 3.
    preE1Index = 0

    for j in range(sentenceCount - 1):
        sentenceText = sentences[j]
        sentOffset = sentenceOffsets[j]
        sentLength = len(sentenceText)
        sentEnd   = sentOffset + sentLength
        
        nextSentText = sentences[j+1]
        nextSentOffset = sentenceOffsets[j + 1]
        nextSentEnd = nextSentOffset + len(sentences[j + 1])

        e1Index = preE1Index

        while e1Index < entityCount:
            e1 = entityNodes[e1Index]
            e1UniprotIds, e1Offset, e1Length, notValidNode = extractAnnotationNode(e1)
            if notValidNode:
                e1Index += 1
                continue
            if e1Offset > sentEnd or e1Offset < sentOffset:
                preE1Index = e1Index
                break
            # offset of crrent entity's tail char.
            e1start = e1Offset- sentOffset
            e1end   = e1start + e1Length

            e2Index = e1Index + 1
            while e2Index < entityCount:
                e2 = entityNodes[e2Index]
                e2UniprotIds, e2Offset, e2Length, notValidNode = extractAnnotationNode(e2)
                if notValidNode:
                    e2Index += 1
                    continue
                # offset of crrent entity's head char.
                if e2Offset < nextSentOffset:
                    e2Index += 1
                    continue
                if e2Offset > nextSentEnd:
                    break
                e2start = e2Offset - nextSentOffset
                e2end  = e2start + e2Length

                for e1Id in e1UniprotIds:
                    for e2Id in e2UniprotIds:
                        # If there is no UniProt id
                        # or the two entities are one.
                        # Then continue.
                        if e1Id == e2Id:
                            continue
                        e1Text = e1.find("text").text.lower()
                        e2Text = e2.find("text").text.lower()
                        # label
                        label = "0"
                        pair = e1Id + "_" + e2Id
                        if pair in labelPairs:
                            label = "1"
                        # swap the mention to UniprotId.
                        sentence = "{} $1 {} {} $2 {}".format(sentenceText[0:e1start], sentenceText[e1end:], nextSentText[0:e2start], nextSentText[e2end:]).lower()
 
                        sentencePin = "$1 {} {} $2".format(sentenceText[e1end:], nextSentText[0:e2start]).lower()
                        if e1Text in sentencePin or e2Text in sentencePin:
                            continue
                        for entity in entityNodes:
                            sentence = sentence.replace("{}".format(entity.find("text").text.lower()), " gene0 ")

                        sentence = sentence.replace("[", ' ')
                        sentence = sentence.replace("]", ' ')
                        sentence = sentence.replace("\"", ' ')
                        sentence = sentence.replace("'", ' ')
                        sentence = sentence.replace(",", ' ')
                        sentence = sentence.replace(".", ' ')
                        sentence = sentence.replace(";", ' ')
                        sentence = sentence.replace(":", ' ')
                        sentence = sentence.replace("*", ' ')
                        sentence = sentence.replace("(", ' ')
                        sentence = sentence.replace(")", ' ')

                        sentence = sentence.replace("+", " ")
                        sentence = sentence.replace("-", ' ')
                        sentence = sentence.replace("--", ' ') 
                        sentence = sentence.replace("/", ' ')    
                        sentence = sentence.replace(">", " ")
                        sentence = sentence.replace(" mg ", " ") 
                        sentence = sentence.replace(" kg ", " ")

                        sentence = re.sub("^that", '', sentence)
                        sentence = re.sub("that$", '', sentence)
                        sentence = re.sub(" that ", ' ', sentence)
                        sentence = re.sub("^the", '', sentence)
                        sentence = re.sub("the$", '', sentence)
                        sentence = re.sub(" the ", ' ', sentence)
                        sentence = re.sub(' [0-9][0-9]*%', ' NUMBER', sentence)
                        sentence = re.sub(' [0-9][0-9]* ', ' NUMBER ', sentence)
                        sentence = re.sub(' [0-9][0-9]* ', ' NUMBER ', sentence)
                        sentence = sentence.strip()
                        sentence = re.sub(' +', ' ', sentence)
                        words = sentence.split(" ")
                        e1Pos = words.index("$1")
                        e2Pos = words.index("$2")
                        startPos = max(0, e1Pos - win)
                        endPos = min(e2Pos + win + 1, len(words))
                        sentence = " ".join(words[startPos:endPos])

                        sentLength = len(sentence.split(" "))
                        if sentLength > 3:
                            formatCorpusStream.write("\t".join([pmid, sentence, e1Id, e2Id, e1Text, e2Text, str(sentLength), label]) + "\n")
                e2Index += 1
            e1Index += 1

def gene2protein(dictPath):
    gene2proteinDict = {}
    with open(dictPath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            if items[1] not in gene2proteinDict:
                gene2proteinDict[items[1]] = []
            gene2proteinDict[items[1]].append(items[0])
    return gene2proteinDict

labelPath = "../data/trainGold.txt"
# one xml file for one document
corpusRoot = "../data/splitOffset/"
samplePath = "../data/xml2InterSentence.txt"
win = 3

parser = argparse.ArgumentParser()
parser.add_argument("--label", default=labelPath, help="path of labels")
parser.add_argument("--offsets", default=corpusRoot, help="path of PubTator format files.")
parser.add_argument("--output", default=samplePath, help="path of sentences.")
parser.add_argument("--win", default=3, type=int)
args = parser.parse_args()

labelPath  = args.label
corpusRoot = args.offsets
samplePath = args.output
win        = args.win

labels = {}
if labelPath:
    with open(labelPath, 'r') as labelsStream:
        for line in labelsStream:
            items = line.strip().split("\t")
            if items[0] not in labels:
                labels[items[0]] = []
            
            entities1 = items[1].split("|")
            entities2 = items[2].split("|")
            for entity1 in entities1:
                for entity2 in entities2:
                    labels[items[0]].append(entity1 + "_" + entity2)
                    labels[items[0]].append(entity2 + "_" + entity1)

formatCorpusStream = open(samplePath, 'w')
d = Path(corpusRoot)

count = 0
for item in d.iterdir():
    print("processing " + item.name)
    tre = ET.parse(str(item))
    root = tre.getroot()

    documentIndex = 3
    for documentNode in root[documentIndex:]:
        count += 1
        pmid = documentNode[0].text
        if pmid not in labels:
            labelPairs = []
        else:
            labelPairs = labels[pmid]
        # skip the id and totle nodes.
        fulltextElements = documentNode[1:]
        for fulltextElement in fulltextElements:
            if fulltextElement.tag == "passage":
                writePassage(fulltextElement, labelPairs, pmid, formatCorpusStream)
print(count)