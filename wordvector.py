import numpy as np
import pandas as pd
import time

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from collections import defaultdict
from tqdm.auto import tqdm

import torch
from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel

def add_period(text):
    text[-1] += "."
    return text

def corpus_by_date(corpus, date):
    grouped_corpus = [add_period(corpus[0])]; length = len(corpus); day = 0
    for i in range(1,length): 
        if date[i] == date[i-1]:
            grouped_corpus[day].extend(add_period(corpus[i]))
        else:
            day = day + 1
            grouped_corpus.append(add_period(corpus[i]))      
    return grouped_corpus

def clean_corpus(corpus):
    before = len(corpus)
    
    cleaned_corpus = []
    for sublist in corpus:
        cleaned_sublist = []
        for item in sublist:
            cleaned_item = ''.join([char for char in item if char.isalpha() or char == '-'])
            if cleaned_item:
                cleaned_sublist.append(cleaned_item)
        if cleaned_sublist:
            cleaned_corpus.append(cleaned_sublist)
    
    after = len(corpus)
    if before != after:
        print("{} item(s) are removed".format(after-before))
    
    return cleaned_corpus
    
def get_vocabulary(corpus, min_occurrence):
    all_words = [word for sublist in corpus for word in sublist]
    word_counter = Counter(all_words)
    vocab = [word for word, count in word_counter.items() if count >= min_occurrence]
    vsize = len(vocab)
    return vocab, vsize

def get_co_occurrence_matrix(corpus,min_occurrence):
    
    vocabulary, v_size = get_vocabulary(corpus, min_occurrence)
    
    word2ind = {}
    for idx, word in enumerate(vocabulary):
        word2ind[word] = idx
    
    E = np.array(np.zeros(v_size*len(corpus)).reshape(v_size,len(corpus)), dtype=np.int8)
    progress_bar = tqdm(range(v_size))
     
    for w,word in enumerate(vocabulary):
        exist = [word in sublist for sublist in corpus]
        E[w,:] = [int(value) for value in exist]
        progress_bar.update(1)
    E = csr_matrix(E)
    M = E@E.T
 
    return M, word2ind

def get_co_occurrence_matrix2(corpus,min_occurrence):
    
    vocabulary, v_size = get_vocabulary(corpus, min_occurrence)
    
    word2ind = {}
    for idx, word in enumerate(vocabulary):
        word2ind[word] = idx
    
    M = np.array(np.zeros(v_size*v_size).reshape(v_size,v_size), dtype=np.int32)
    progress_bar = tqdm(range(len(corpus)))
 
    for sentence in corpus:
        l = len(sentence)
        for i in range(l):
            if sentence[i] in vocabulary:
                for j in range(l):
                    if sentence[j] in vocabulary:
                        idx1 = word2ind[sentence[i]]
                        idx2 = word2ind[sentence[j]]
                        M[idx1,idx2] = M[idx1,idx2] + 1
        progress_bar.update(1)
    
    return M, word2ind

def reduce_dimension(M, k=2):

    n_iter = 10
    random_state = 595
    
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state)
    X = svd.fit_transform(M)
    for i in range(X.shape[0]):
        scaler = np.sqrt(sum([x**2 for x in X[i,:]]))
        X[i,:] = X[i,:]/scaler
    
    M_reduced = X
    return M_reduced
    
def get_k_vectors(corpus,M,word2ind):
    
    kvec = pd.DataFrame(columns=np.array(range(M.shape[1])))
    progress_bar = tqdm(range(len(corpus)))
    
    for sentence in corpus:
        vector = np.zeros(M.shape[1]);
        for word in sentence:
            if word in word2ind.keys():
                vector = vector + M[word2ind[word],:]
        kvec.loc[len(kvec),:] = vector / len(sentence)
        
        progress_bar.update(1)
        
    return kvec


def get_v_vectors(corpus):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
   
    vectors = []
    progress_bar = tqdm(range(len(corpus)))

    for sentence_list in corpus:
        text = " ".join(sentence_list)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        sentence_vector = torch.mean(last_hidden_states, dim=1)
        sentence_vector = sentence_vector.squeeze()
        vectors.append(sentence_vector)
        
        progress_bar.update(1)
            
    vectors = torch.stack(vectors)
    vvec = pd.DataFrame(vectors.numpy())
    return vvec