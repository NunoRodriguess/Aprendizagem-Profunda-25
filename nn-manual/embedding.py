import numpy as np

def load_glove_embeddings(file_path, embedding_dim=100):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
    return embeddings_index

def create_embedding_matrix(word_index, embeddings_index, embedding_dim=100):
    vocab_size = len(word_index) + 1  # +1 for padding if needed
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Use pre-trained vector
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))  # Random init

    return embedding_matrix

def features_to_word_index(vocab):
    return {word: i for i, word in enumerate(vocab)}

def convert_onehot_to_indices(onehot_matrix):
    return np.argmax(onehot_matrix, axis=1)

def load_glove_embeddings_matrix(file_path, word_index, embedding_dim=100):
    """ Load GloVe vectors and create an embedding matrix """
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector
    
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    return embedding_matrix
