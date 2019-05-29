#!/usr/bin/python3
import argparse
import xml.etree.ElementTree as ET
# def gene2protein(dictPath):
#     gene2proteinDict = {}
#     with open(dictPath, 'r') as f:
#         for line in f:
#             items = line.strip().split("\t")
#             if items[1] not in gene2proteinDict:
#                 gene2proteinDict[items[1]] = []
#             gene2proteinDict[items[1]].append(items[0])
#     return gene2proteinDict

#mapping = "/home/laboratory/lab/BioCreative/codePlayer/KBExtractor/KB/geneIdToUniprotId.txt"
labelPath = "./train_label.txt"
corpus  = "/home/laboratory/lab/ACL2017/corpus/BioCreative/trainOffset/PMtask_Relations_TrainingSet.xml"

# labelPath = "/home/laboratory/lab/ACL2017/corpus/BioCreative/gold/label_oldTestset.txt"
# corpus  = "/home/laboratory/lab/ACL2017/corpus/BioCreative/gold/PMtask_Relation_TestSet_updated.xml"

parser = argparse.ArgumentParser()
parser.add_argument("--label", default=labelPath, help="path of labels")
parser.add_argument("--corpus", default=corpus, help="path of PubTator format files.")
#parser.add_argument("--mapping", default=mapping, help="path to gene id to protein id file.")
args = parser.parse_args()

# "/home/laboratory/lab/BioCreative/2010/BC2/bc2_ips_pmid2ppi_train.txt"
# "/home/laboratory/lab/BioCreative/2010/BC2/trainOffset/"
# "/home/laboratory/lab/BioCreative/2010/BC2/corpus_train.txt"
labelPath  = args.label
corpus = args.corpus
#mapping    = args.mapping

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
