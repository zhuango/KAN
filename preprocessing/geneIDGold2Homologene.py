mappingFile = "/home/laboratory/lab/ACL2017/corpus/BioCreative/gold/homologene.data"
goldFile = "/home/laboratory/lab/ACL2017/corpus/BioCreative/gold/gold.txt"
homoGold = "/home/laboratory/lab/ACL2017/corpus/BioCreative/gold/gold_homo.txt"
mapping = {}
with open(mappingFile, 'r') as f:
    for line in f:
        items = line.strip().split("\t")
        mapping[items[2]] = items[0]
homogoldSet = set()
with open(goldFile, 'r') as f:
    for line in f:
        items = line.strip().split("\t")
        if(items[1] in mapping and items[2] in mapping):
            homogoldSet.add((items[0], mapping[items[1]], mapping[items[2]]))
        else:
            print(items[1], items[2])
            homogoldSet.add((items[0], items[1], items[2]))
with open(homoGold, 'w') as f:
    for tup in homogoldSet:
        f.write("\t".join(tup) + "\n")
    