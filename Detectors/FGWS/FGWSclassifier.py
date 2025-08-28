# functions to train, test FGWS
# Example:

import sys
import os
import csv
sys.path.append('../../Classifiers/GooglePerspective/')
sys.path.append('../../Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel
import math
import pickle
from nltk.corpus import wordnet
from polyglot.mapping import Embedding
import re
import spacy
from spacy.lang.en import English


class FGWSDetector:
    
    def __init__(self, freq_file = '', syn_file = '', gamma = 0.2, delta = 2.0, class_alg = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-ag-news'):
        self.freqs = {}
        self.syns = {}
        

        # read in synonyms and frequencies
        freqcsv = csv.reader(open(freq_file), delimiter = '\t')
        for cur in freqcsv:
            self.freqs[cur[0]] = float(cur[1])


        syncsv = csv.reader(open(syn_file), delimiter = '\t')
        for cur in syncsv:
            self.syns[cur[0]] = cur[1:]

        self.gamma = float(gamma)
        self.delta = float(delta)

        
        nlp = English()
        self.spacy_tokenizer = nlp.tokenizer
        if(class_alg == 'Perspective'):
            self.classifier = Perspective(threshold = 0.5, select = persp_key)
            #self.tokenizer = None
        elif(class_alg == 'HuggingFaceModel'):
            self.classifier = HuggingFaceModel(hfmodel)
            #self.tokenizer = AutoTokenizer.from_pretrained(hfmodel)


    def predict(self, test_query, attacked_class = 0):
        new_text = clean_str(test_query, self.spacy_tokenizer)
        # replace each low freq word with high freq syn
        
        
        for i in range(len(new_text)):
            word = new_text[i]
            if(word not in self.freqs):#add in for possible replacement
                self.freqs[word] = 0 
                self.syns[word] = list(set(get_word_net_synonyms(word) + get_embedding_nns(word, 'glove.42B.300d.txt')))

            if(self.freqs[word] < self.delta):
                # try and replace with more frequent word
                highest_freq = 0
                replace_syn = word

                for syn in self.syns[word]:
                    # choose syn which has highest frequency
                    if(syn in self.freqs and self.freqs[syn] >= self.delta and self.freqs[syn] > highest_freq):
                        highest_freq = self.freqs[syn]
                        replace_syn = syn

                new_text[i] = replace_syn


        orig_pred, _ = self.classifier.predict(test_query)
        _, orig_prob = self.classifier.predict(test_query, orig_pred)
        _, new_prob = self.classifier.predict(' '.join(new_text), orig_pred)
        
        if(abs(orig_prob - new_prob) > self.gamma):
            pred = 1 #attack
        else:
            pred = 0 #not attack

        prob = orig_prob - new_prob 
        return pred, prob


    # allow multiple queries in form of list
    def predictMultiple(self, test_queries, attacked_class = 0):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test, attacked_class)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs





def get_embedding_nns(word, embeddings):
    try:
        neighbors = embeddings.nearest_neighbors(word)
    except: #if replacement word does not exist in the vocabulary 
        neighbors = []

    return neighbors



# from https://github.com/maximilianmozes/fgws/blob/main/utils.py needed for reproducing FGWS results
def clean_str(string, tokenizer=None):
    """
    Parts adapted from https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/mydatasets.py
    """
    assert isinstance(string, str)
    string = string.replace("<br />", "")
    string = re.sub(r"[^a-zA-Z0-9.]+", " ", string)

    return (
        string.strip().lower().split()
        if tokenizer is None
        else [t.text.lower() for t in tokenizer(string.strip())]
    )


# from https://github.com/maximilianmozes/fgws/blob/main/utils.py needed for reproducing FGWS results
def get_word_net_synonyms(word):
    """
    Parts from https://github.com/JHL-HUST/PWWS/blob/master/paraphrase.py
    """
    synonyms = []

    for synset in wordnet.synsets(word):
        for w in synset.lemmas():
            synonyms.append(w.name().replace("_", " "))

    synonyms = sorted(
        list(set([x.lower() for x in synonyms if len(x.split()) == 1]) - {word})
    )

    return synonyms



def calculateFrequenciesAndSynonyms(trainfile):#, tokenizer=None):
    # process training/test data
    traincsv = csv.reader(open(trainfile, 'r'), delimiter ='\t')
    fname = os.path.basename(trainfile)
    freqcsv = csv.writer(open('frequencies_' + fname, 'w'), delimiter = '\t')
    syncsv = csv.writer(open('synonyms_' + fname, 'w'), delimiter = '\t')


    train_x = []

    freqs = {}
    syns = {}

    nlp = English()
    spacy_tokenizer = nlp.tokenizer
    
    # read in training data
    for cur in traincsv:

        if(cur[0].isdigit()):
            cur = cur[1:] # remove idx 

        text = clean_str(cur[0], spacy_tokenizer)
        for word in text:
            if(word not in freqs):
                freqs[word] = 0
                syns[word] = []

            freqs[word] += 1
        

            

    # apply nat log for the counts
    for word in freqs:
        freqs[word] = math.log(freqs[word])



    # sort and output freqs to frequency file
    for word in sorted(freqs.keys()):
        outrow = [word, freqs[word]]
        freqcsv.writerow(outrow)
        


    embeddings = Embedding.from_glove('glove.42B.300d.txt')
    # get synonyms for each word
    for word in syns:
        wn_syns = get_word_net_synonyms(word)
        emb_syns = get_embedding_nns(word, embeddings)
        syns[word] = list(set(wn_syns + emb_syns))
        outrow = [word]
        outrow.extend(syns[word])

        syncsv.writerow(outrow)



def detectAttacks(testfile, freq_file, syn_file, delta = 2, gamma = 0.2, class_alg = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-ag-news'):

    # set up FGWS detector
    detector = FGWSDetector(freq_file, syn_file, gamma = float(gamma), delta = float(delta), class_alg = class_alg, hfmodel = hfmodel)

    # read in/test data
    testcsv = csv.reader(open(testfile), delimiter = '\t')
    
    total = 0
    correct = 0
    for cur in testcsv:
        text = cur[0]
        gold = float(cur[1])

        label, prob = detector.predict(text)
        print(gold, label, prob)

        if(gold == label):
            correct += 1
            
        total += 1
        
    print("Accuracy for", testfile, "-", correct/total * 100.0)


        
            
    

    




if(__name__ == "__main__"):
    if(sys.argv[-1] == 'detect'):
        detectAttacks(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        calculateFrequenciesAndSynonyms(sys.argv[1])#, sys.argv[2])
