from __future__ import print_function
import gensim
from os.path import join, exists, split
import os
import numpy as np


def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in vocabulary_inv.items()}
    return embedding_weights



import data_helpers

print("Loading data...")
x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data()
vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
embedding_weights = train_word2vec(x, vocabulary_inv)

f=open('embedding_weights.txt','w')
f.write(str(embedding_weights))
f.close()
print(len(embedding_weights))

print(x.shape)
print(y.shape)
np.save('x.npy',x)
np.save('y.npy',y)

weights_numpy = np.array([v for v in embedding_weights.values()])
print(weights_numpy.shape)
np.save('embedding_weights.npy',weights_numpy)