# simple module for classifying the test files (perturbed or not)

import sys
sys.path.append('Classifiers/GooglePerspective/')
sys.path.append('Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel
import csv
sys.path.append('Detectors/WDR')
from WDRclassifier import WDRDetector
import os
sys.path.append('Detectors/FGWS')
from FGWSclassifier import FGWSDetector 
sys.path.append('Detectors/UAPAD/UAPAD')
from UAPADclassifier import UAPADDetector


def detect(dataset = 'imdb_1000.tsv', outfile = 'output.tsv', class_alg = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-imdb', detection_method = 'WDR', det_location = 'Detectors/WDR/totalTrain_pwwsVsalbert-base-v2-ag-news_ag-news_train-WDR_RFMODEL.sav', gamma = 0.05, delta = 0.9, delta0_pos = 0, delta1_pos=0, delta2_pos = 0, delta3_pos=0, delta_file_loc = 'results/textattack/distilbert-base-uncased-imdb/imdb', delta_weight = 0.5, num_labels = 2):

    incsv = csv.reader(open(dataset), delimiter = '\t')
    outcsv = csv.writer(open(outfile, 'w'), delimiter = '\t')

        
    if(detection_method == 'WDR'):
        detector = WDRDetector(det_location = det_location, class_alg = class_alg, hfmodel = hfmodel)
    elif(detection_method == 'FGWS'):
            base_loc = os.path.basename(det_location)
            directory = os.path.dirname(det_location)
            detector = FGWSDetector(freq_file = directory + '/frequencies_' + base_loc, syn_file = directory + '/synonyms_' + base_loc, gamma = gamma, delta = delta, class_alg = class_alg, hfmodel = hfmodel)
    elif(detection_method == 'UAPAD'):
            detector = UAPADDetector(hfmodel = hfmodel, delta0_pos = int(delta0_pos), delta1_pos = int(delta1_pos), delta2_pos = int(delta2_pos), delta3_pos = int(delta3_pos), delta_file_loc = delta_file_loc, delta_weight = float(delta_weight), num_labels = int(num_labels))

    
    
    
    # iterate through each text, make a prediction, record, prediction and gold label probability
    for cur in incsv:
        
        if(len(cur) == 2): # non attacked text
            cur_text = cur[0]
            cur_gold_label = cur[1]
            cur_num_queries = 0
        elif(len(cur) == 3):
            if(cur[0].isdigit()): # idx, non attacked
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
        
        pred, prob = detector.predict(cur_text, 1)


        out_row.extend([pred, prob])

        
        
        outcsv.writerow(out_row)





if(len(sys.argv) == 7):
    detect(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
elif(len(sys.argv) == 9):
    detect(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
else:
    detect(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7],sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15])

