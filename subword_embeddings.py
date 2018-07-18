# -*- coding: utf-8 -*-
from gensim.models import KeyedVectors
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("en.wiki.bpe.op1000.model")
data = sp.EncodeAsPieces("This is a test")
print(data)

# load any model saved to disk using gensim keyed vectors.
# remove binary = True argument
model = KeyedVectors.load_word2vec_format("en.wiki.bpe.op3000.d100.w2v.bin",binary=True)

subwords = "▁mel ford shire".split()
print(subwords)
bpe_embs = model[subwords]
print(bpe_embs.shape)
# print(bpe_embs)
print(model.most_similar("shire"))

# print(model.wv.vocab)

# https://radimrehurek.com/gensim/models/keyedvectors.html
# https://radimrehurek.com/gensim/models/word2vec.html
