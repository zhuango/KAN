#!/usr/bin/python3
import argparse
import xml.etree.ElementTree as ET

labelPath = "../data/trainGold.txt"
corpus  = "../data/PMtask_Relations_TrainingSet.xml"

parser = argparse.ArgumentParser()
parser.add_argument("--label", default=labelPath, help="path of labels")
parser.add_argument("--corpus", default=corpus, help="path of PubTator format files.")
args = parser.parse_args()

labelPath  = args.label
corpus = args.corpus

geneSet = set()

labelStream = open(labelPath, 'w')
tre = ET.parse(str(corpus))
root = tre.getroot()
documentIndex = 3
documentNode = root[documentIndex]

for docNode in root[documentIndex:]:
    pmid = docNode[0].text
    relations = docNode
    hasRelation = False
    for relation in relations:
        if(relation.tag != "relation"):
            continue
        geneId1 = relation[0].text
        geneId2 = relation[1].text
        hasRelation = True
        if "{}\t{}\t{}\n".format(pmid, geneId1, geneId2) not in geneSet:
            labelStream.write("{}\t{}\t{}\n".format(pmid, geneId1, geneId2))
            geneSet.add("{}\t{}\t{}\n".format(pmid, geneId1, geneId2))
            geneSet.add("{}\t{}\t{}\n".format(pmid, geneId2, geneId1))
    if hasRelation:
        print(pmid)
print(geneSet)
labelStream.close()
