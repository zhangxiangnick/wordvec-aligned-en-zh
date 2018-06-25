import numpy as np

def read_wordvec(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        n = int(f.readline().split(' ')[0])
        words = []
        vectors = []
        word2id = {}
        for i in range(n):
            word, vec = f.readline().split(' ', 1)
            words.append(word)
            vectors.append(np.fromstring(vec, sep=' '))
            word2id[word] = i
        vectors = np.vstack(vectors)
    return words, vectors, word2id
