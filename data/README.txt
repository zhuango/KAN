There are two folders in this directory:
KB:
    entity2id.txt: mapping from Entrez Gene ID to number during training TransE.
    entity2id.vec: entity embeddings learned from TransE.
    relation2vec.vec: relation embeddings learned from TransE.
    triple2id.txt: [entity1 number]  [entity2 number] [relation number]
wordEmb:
    bio-embed100: word embeddings
    bio-word2id100: mapping from word to number.

There is file named trainingSample.txt: example format of training instance.
Format of instance:
[PMID]  [Context]   [Entrez Gene ID of entity1]    [Entrez Gene ID of entity2]  [Mention of entity1]    [Mention of entity2]    [Length of context] [Label: 1 for postive and 0 for negative]
