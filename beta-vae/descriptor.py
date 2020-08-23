"""
ENVIRONMENT - SneakerFinder_Dev) formerly SnkrSL  


This script processes the descriptions. text2sneak.py requires files made by this sciprt

It is broken up into 5 subparts:
    1. Loading in spacy and exploring it's capabilities
    2. Loading in the data (picture, description) from the scraped  DB
    3. ?? sneaker class that takes in various information about an object and stores it so that it can be sampled for randomized descriptions later.
    5. Generate the descriptions, save them to file, and inspect the descriptions for quality.
"""

#%% Imports
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pandas as pd
import random
from tqdm import tqdm
import random as rn

# import inflect
import spacy

# import en_core_web_lg
import displacy

# from nltk.corpus import wordnet as wn
import math

import utils as ut
import configs as cf


np.set_printoptions(precision=3, suppress=True)

#%% Setup text processing and load in dfmeta
# inflect = inflect.engine()
vocab = set()
corpdict = {}

# cats_to_load = ['Table','Chair','Lamp','Faucet','Clock','Bottle','Vase','Laptop','Bed','Mug','Bowl']
# catids_to_load = [4379243,3001627,3636649,3325088,3046257,2876657,3593526,3642806,2818832,3797390,2880940]

dfdesc = pd.read_csv(cf.META_DATA_CSV)

#%% Setup spacy and some related methods
def get_embeddings(vocab):
    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype="float32")
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


def most_similar(word):
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.has_vector]
    # queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15 and w.has_vector]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return by_similarity[:10]


def closestVect(vect, num_to_get=10):
    queries = [w for w in nlp.vocab if w.has_vector]
    # queries = [w for w in nlp.vocab if w.prob >= -15 and w.has_vector]
    by_similarity = sorted(
        queries, key=lambda w: cosine_similarity(w.vector.reshape(1, -1), vect.reshape(1, -1)), reverse=True
    )
    return by_similarity[:num_to_get]


#%%  get vocab
# load the "model"
nlp = spacy.load("en_core_web_md")

for orth in nlp.vocab.vectors:
    _ = nlp.vocab[orth]

#%%  get vocab

nlp = en_core_web_lg.load()

# remove the empty placeholder prob table
if nlp.vocab.lookups_extra.has_table("lexeme_prob"):
    nlp.vocab.lookups_extra.remove_table("lexeme_prob")

# access any `.prob` to load the full table into the model
assert nlp.vocab["a"].prob == -3.9297883511


# n_vectors = 105000  # number of vectors to keep
# removed_words = nlp.vocab.prune_vectors(n_vectors)

# assert len(nlp.vocab.vectors) <= n_vectors  # unique vectors have been pruned
# assert nlp.vocab.vectors.n_keys > n_vectors  # but not the total entries
# embeddings = get_embeddings(nlp.vocab)


#%% Generate all descriptions and stack them into a numpy array
# all_descs = []
# for row in tqdm(dfdesc.iterrows()):
#     all_descs.append([row.hero_fullpath, row.description])

all_descs = [list(z) for z in zip(dfdesc["hero_fullpath"], dfdesc["description"])]


# dnp = dfdesc["description"].values
dnp = np.stack(all_descs)


#%% Get word count stats for the generated descriptions
total = 0
maxlen = 0
for d in dnp[:, 1]:
    numwords = len(d.split())
    total += numwords
    maxlen = max(numwords, maxlen)

num_unique_descs = np.unique(dnp[:, 1]).shape[0]
print(
    "Average words per desc: {:.1f} \nMax words: {} \nUnique descs: {} / {}  = {:.1f}%".format(
        total / len(dnp[:, 1]), maxlen, num_unique_descs, len(dnp), 100 * num_unique_descs / len(dnp)
    )
)

#%% Save descriptions to file
np.save(os.path.join(cf.DATA_DIR, "alldnp.npy"), dnp, allow_pickle=True)


#%% testing tools tagging testing
##############TESTING ####################

#%% Seeing what the closest vector is in vector space
cvs = closestVect(nlp("sneaker")[0].vector)
for w in cvs:
    print(w.text)

#%% Using the builtin spacy method to do the same as above for comparison
syns = most_similar(nlp.vocab["sneaker"])
for w in syns:
    print(w.text, w.cluster)

text = nlp("a large thing")

# dfdesc.description[0]
xx = "released in december   the air jordan 11 retro ‘bred’ 2012 brings back the famous blackred colorway to the jordan 11 silhouette the design features a black mesh upper with black patent leather overlays   red jumpman accents   a white midsole   and a translucent red outsole originally released in 1995   the sneaker was retro’d in 2001 and in 2008 alongside an air jordan 12 as part of the ‘collezione pack’"
text = nlp(xx)

for token in text:
    print(
        "{:12}  {:5}  {:6}  {}".format(
            str(token), str(token.tag_), str(token.pos_), str(spacy.explain(token.tag_))
        )
    )

print("\n\n")
for token in text:
    print(
        "{:10}  {:8}  {:8}  {:8}  {:8}  {:15}  {}".format(
            token.text,
            token.dep_,
            token.tag_,
            token.pos_,
            token.head.text,
            str([item for item in token.conjuncts]),
            [child for child in token.children],
        )
    )

# for item in text:
#     print(
#         "{:10}  {}".format(
#             item.text,
#             [
#                 syn.name().split(".")[0]
#                 for syn in wn.synsets(item.text, pos=wn.ADJ)[:10]
#                 if not syn.name().split(".")[0] == item.text
#             ],
#         )
#     )
# wn.synset("tall.a.01").lemmas()[0].antonyms()


#%% Shape description methods
# TODO:   re-summarize

"""
These methods take in a row of the dfmeta file and populate a standardized set of items that the shape class
then uses to generate randomized descriptions. It may use any subset of the available items.

From dfmeta these methods get :
    1. subcat       : The subcategory. For example, a chair may specfically be a swivel chair.
    2. sdescriptors : Shape descriptors. Changes for each category but could be like tall for chairs or thick for clocks. Every category method is preceded by a ...sizes lists which defines which axes correspond to which shape descriptions.
    3. gdescriptors : General descriptors. For example, could be a decorative or jointed lamp.
    4. spnames      : Sub part names. A listing of all interesting sub parts contained in the tree. Interesting defined as relatively rare.
    5. negatives    : Things that this object doesn't have that other similar objects often do.
    6. contains     : If it contains anything. Only relevant for a few categories like vase which could contain a plant or soil or a bowl that could be full or empty.
"""


#%% Helper methods
def clamp(value, minv, maxv):
    return int(max(min(value, maxv), minv))


def getFreq(name):
    if name not in corpdict:
        return 0
    return corpdict[name]


# def rarestCat(row, min_cutoff=10, exclude="NONE"):
#     subcats = row.subcats.split(",")
#     cat_freq = [
#         getFreq(cat) if int(getFreq(cat)) >= min_cutoff or cat == exclude else 999999 for cat in subcats
#     ]
#     if min(cat_freq) >= 999999:
#         return row.cattext.lower()
#     return subcats[np.argmin(cat_freq)]


def joinPhrases(phrases):
    phrases = [p for p in phrases if not p == ""]
    return ". ".join(phrases)


# def multiScriptor(thing, qty, many_thresh=7):
#     if qty == 0:
#         return "no {} ".format(inflect.plural(thing))
#     for i in range(1, many_thresh + 1):
#         if qty == i:
#             return "{} {} ".format(inflect.number_to_words(i), inflect.plural(thing, count=i))
#     return "many {} ".format(inflect.plural(thing))


# def getSumDet(det, name):
#     count = sum([int(item[3]) for item in det if item[0] == name])
#     return count


# def printDetArr(det, max_level=10):
#     for item in det:
#         name, level, children, quantity = (item[0]).lower(), int(item[1]), int(item[2]), int(item[3])
#         if level > max_level:
#             continue
#         freq = corpdict[name]
#         print(
#             "{:1d} :{:2d} :{:2d} : {:5d} : {}{}".format(
#                 level, children, quantity, freq, "  " * int(level + 1), name
#             )
#         )


# The details array contains: [self.name, level, children, quantity]
def detToArr(details):
    dets = [[item.strip() for item in d.split(",")] for d in (" " + details).split("|")[:-1]]
    dets = np.array(dets)
    return dets


def listSubcats(catid):
    allsubcats = []
    dffilter = dfdesc[dfdesc.cattext == cats_to_load[catid]]
    for i, row in dffilter.iterrows():
        allsubcats.extend(row.subcats.split(","))
    for item in sorted(set(allsubcats)):
        freq = 0
        try:
            freq = corpdict[item]
        except:
            freq = "N/A"
        print("{:5}   {}".format(freq, item))


def deleteSentences(row, rate=0.2):
    desc = row.desc.values[0]
    s = desc.split(".")[:-1]
    for sentence in s[1:]:
        if rn.uniform(0, 1) < rate:
            s.remove(sentence)

    result = " . ".join(s) + " . "
    row.desc = result
    return row


def shuffleSentences(row):
    desc = row.desc.values[0]
    s = desc.split(".")[:-1]
    rn.shuffle(s)
    result = " . ".join(s) + " . "
    row.desc = result
    return row


fix_replacements = [[".", " . "], [",", " , "], ["  ", " "], ["   ", " "]]


def fixPuncs(row):
    desc = row.desc.values[0]
    for rp in fix_replacements:
        desc = desc.replace(rp[0], rp[1])
    row.desc = desc
    return row


def buildCorpus(dfdesc):
    global corpdict
    corpus = []
    for index, row in tqdm(dfdesc.iterrows(), total=len(dfdesc)):  # .iloc[:1000]
        try:
            corpus.extend([str(row.description).lower()])
            corpus.extend([word.lower() for word in row.subcats.split(",")])
            dets = [item.split(",")[0].strip().lower() for item in row.details.split("|")[:-1]]
            corpus.extend(dets)
        except:
            pass

    phrases = Counter(corpus).keys()
    counts = Counter(corpus).values()
    corpdict = {k: v for k, v in zip(phrases, counts)}

    for p, c in zip(phrases, counts):
        print("{:25}    {}".format(p, c))

    corpdict = {k: v for k, v in sorted(corpdict.items(), key=lambda item: item[1])}
    return corpdict


if len(corpdict) < 5:
    buildCorpus(dfdesc)


# #%% Setup class balancer. These numbers are based on estimates for how much attention each category requires relative to how many samples are available.
# balancer = {
# 'Table' : 1,
# 'Chair' : 1.4,
# 'Lamp' : 1.5,
# 'Faucet' : 1.8,
# 'Clock' : 1.3,
# 'Bottle' : 1.4,
# 'Vase' : 1.4,
# 'Laptop' : 1.2,
# 'Bed' : 3.5,
# 'Mug' : 2.5,
# 'Bowl' : 2.9 }

#
#%% Inspect some objects and their generated descriptions
# catid = 0

# while True:
#     index = rn.randint(1, len(dfdesc) - 1)
#     row = dfdesc.iloc[index]
#     if not row.cattext == cats_to_load[catid]:
#         continue
#     print(
#         "-------------------------------------------{}------------------------------------------------".format(
#             index
#         )
#     )
#     print("{:20}    {}".format(row.cattext, row.mid))

#     sh = dRow(row)
#     print(sh)
#     for i in range(int(sh.getComplexity() * balancer[row.cattext] * 1)):
#         print(sh.getDesc())

#     ut.showPic(dfdesc.iloc[index].mid, title=index)
#     try:
#         i = input("")
#         if i == "s":
#             print("showing binvox...")
#             ut.showBinvox(row.mid)
#     except (KeyboardInterrupt, SystemExit):
#         break

