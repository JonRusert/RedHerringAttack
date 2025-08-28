# simple module for classifying the test files (perturbed or not)

import sys
sys.path.append('Classifiers/GooglePerspective/')
sys.path.append('Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel
import csv




def classify(dataset = 'imdb_1000.tsv', outfile = 'output.tsv', class_alg = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-imdb'):

    incsv = csv.reader(open(dataset), delimiter = '\t')
    outcsv = csv.writer(open(outfile, 'w'), delimiter = '\t')

        

    # initialize classifier
    if(class_alg == 'Perspective'):
        classifier = Perspective(threshold = 0.5, select = persp_key)
    elif(class_alg == 'HuggingFaceModel'):
        classifier = HuggingFaceModel(hfmodel)


    
    
    # iterate through each text, make a prediction, record, prediction and gold label probability
    for cur in incsv:
        
        if(len(cur) == 2):#non query test texts (non attacked)
            cur_text = cur[0]
            cur_gold_label = cur[1]
            cur_num_queries = 0
        elif(len(cur) == 3):
            if(cur[0].isdigit()): # idx
                cur_text = cur[1]
                cur_gold_label = cur[2]
                cur_num_queries = 0
            else:
                cur_text = cur[0]
                cur_gold_label = cur[1]
                cur_num_queries = cur[2]
        elif(len(cur) == 4): # for nli tasks with 2 sentences,note idx not present in attack files (but present in dataset files)
            if(cur[0].isdigit()): # if input has idx in first column
                cur_text = cur[1]
                cur_gold_label = cur[2]
                cur_num_queries = cur[3]
            else: # nli task
                cur_text = (cur[0], cur[1])
                cur_gold_label = cur[2]
                cur_num_queries = cur[3]
        else: #nli task with idx
            cur_text = (cur[1], cur[2])
            cur_gold_label = cur[3]
            cur_num_queries = cur[4]
            

        if(type(cur_text) == tuple): # for nli tasks
            out_row = [cur_text[0], cur_text[1], cur_gold_label, cur_num_queries]
        else:
            out_row = [cur_text, cur_gold_label, cur_num_queries] 
        

        # avoid problems of labels being '1.0' or '1'
        cur_gold_label = int(float(cur_gold_label))
        pred, prob = classifier.predict(cur_text, cur_gold_label)


        out_row.extend([pred, prob])

        
        
        outcsv.writerow(out_row)








classify(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
 

