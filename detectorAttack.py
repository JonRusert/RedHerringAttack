# similar to redherring atttack but only focuses on the detectors and does not care about the classifier

import sys
import os
sys.path.append('Classifiers/GooglePerspective/')
sys.path.append('Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel
import csv
from transformers import pipeline
sys.path.append('Detectors/WDR')
sys.path.append('Detectors/FGWS')
sys.path.append('Detectors/UAPAD/UAPAD')
from WDRclassifier import WDRDetector
from FGWSclassifier import FGWSDetector 
from UAPADclassifier import UAPADDetector


class DetectorAttack:


    def __init__(self, obf_method = 'GS_MR', class_alg = 'Perspective', persp_key = 1, model_location = 'textattack/bert-base-uncased-imdb', detection_method = 'RSV', det_location = 'Detectors/WDR/totalTrain_pwwsVsalbert-base-v2-ag-news_ag-new\|s_train-WDR_RFMODEL.sav', gamma = 0.05, delta = 0.9, delta0_pos = 0, delta1_pos=0, delta2_pos = 0, delta3_pos=0, delta_file_loc = 'results/textattack/distilbert-base-uncased-imdb/imdb', delta_weight = 0.5, num_labels = 2):
        self.obf_method = obf_method
   
        if(detection_method == 'WDR'):
            self.detector = WDRDetector(det_location = det_location, class_alg = class_alg, hfmodel = model_location)
        elif(detection_method == 'FGWS'):
            base_loc = os.path.basename(det_location)
            directory = os.path.dirname(det_location)
            self.detector = FGWSDetector(freq_file = directory + '/frequencies_' + base_loc, syn_file = directory + '/synonyms_' + base_loc, gamma = gamma, delta = delta, class_alg = class_alg, hfmodel = model_location)
        elif(detection_method == 'UAPAD'):
            self.detector = UAPADDetector(hfmodel = model_location, delta0_pos = delta0_pos, delta1_pos = delta1_pos, delta2_pos = delta2_pos, delta3_pos = delta3_pos, delta_file_loc = delta_file_loc, delta_weight = delta_weight, num_labels = num_labels)
        self.query_count = 0
        self.infiller = pipeline("fill-mask", "bert-base-uncased")


    # takes in masked text word the word is to be replaced
    # leverages a MLM to find appropriate replacement
    def MaskReplace(self, query, attack_class, top_k = 3, beg_sentence = None):
        
        print(query)
        #get top options
        options = self.infiller(query, top_k = top_k)

        det_probs = []
        class_probs = []
        sequences = []

        #starting with best, replace mask and test classifier and detection
        for cur_option in options:
            cur_query = cur_option['sequence']

            if(beg_sentence):
                det_pred, det_prob = self.detector.predict((beg_sentence, cur_query), 1) # 1 is attack occuring for detector
            else:
                det_pred, det_prob = self.detector.predict(cur_query, 1)
                
            
            # best case, detector is false flag for attack
            if(int(det_pred) == 1):
                return cur_query, True
            det_probs.append(det_prob)
            sequences.append(cur_query)

        
        if(len(det_probs) != 0):
            scores = [det_probs[i] for i in range(len(det_probs))]
            max_pos = scores.index(max(scores))
            chosen_sequence = sequences[max_pos]

            return chosen_sequence, False

        else: # were not able to change this word without flipping the class, so do not change it
            return query, False




    # beg sentence is there for tasks like nli which take 2 sentences (beg, query) and both are needed to determine which word to drop in query
    def GreedySelect(self, query, attack_class = 0, beg_sentence = None):
        orig_text = query
        #query = self.preProcessText(query)

        # get initial probability for query 
        if(beg_sentence):
            _, initial_prob = self.detector.predict((beg_sentence, query), attack_class)
            self.query_count += 1
        
        else:
            _, initial_prob = self.detector.predict(query, attack_class)
            self.query_count += 1
        

        #print(query, initial_prob)
        needsReplacing = []
        split_query = query.split()
            
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            
            if(beg_sentence): #nli tasks, attack second sentence, but need first for classification
                variations.append((beg_sentence, modified_query))
            else:
                variations.append(modified_query)

        # get probabilities for all variations

        orig_preds, var_probs = self.detector.predictMultiple(variations, attack_class)
        self.query_count += len(variations)
        

        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        



        return prob_diffs

    

        

    

    # GreedySelect MaskReplace
    def GS_MR(self, query, attack_class = 0):
        self.query_count = 0

        first_sentence = None        
        if(type(query) == tuple): # for nli tasks, only attack the response sentence
            first_sentence = query[0]
            query = query[1]

        perturbed_query = query
        orig_pos = {}    
        split_query = query.split()
        full_query = query.split()

        #note original position so when multiple are removed, original is retained
        for i in range(len(split_query)):
            orig_pos[i] = i

        needsReplacing = []
        done = False
    
        # get all gs probs
        all_probs = self.GreedySelect(' '.join(split_query), attack_class, first_sentence)
        remaining_probs = all_probs

        
        while(not done):
            # while
            ## Select Word to replace, replace, then check if tricked classifier, repeat until this does so
            next_replace = remaining_probs.index(max(remaining_probs))


            print(next_replace)
            orig_repl = orig_pos[next_replace]

            print(orig_repl)
            #needsReplacing.append(orig_repl)
            split_query.pop(next_replace)
            remaining_probs.pop(next_replace)

            ## Infill Step
            masked_query = full_query[:orig_repl] + ['[MASK]'] + full_query[orig_repl:]
            print(masked_query)

            perturbed_query, successful = self.MaskReplace(' '.join(masked_query), attack_class, 10, first_sentence)

            if('[MASK]' not in perturbed_query): # maskreplace did modify query
                full_query = perturbed_query.split()
            #print(pert_pred, pert_prob, '\n', perturbed_query)

            if(successful or len(split_query) == 0):
                done = True
            else: # update original positions
                for i in range(len(split_query)):            
                    if(i < next_replace):
                        orig_pos[i] = orig_pos[i]
                    elif(i >= next_replace):
                        orig_pos[i] = orig_pos[i+1]




        
        return perturbed_query, self.query_count






    def obfuscate(self, query, attack_class = 0):
        if(self.obf_method == 'GS_MR'):
            return self.GS_MR(query, attack_class)



def main(dataset = 'imdb.tsv', num_examples = 1000, offset = 0, outfile = 'output.tsv', classifier = 'HuggingFaceModel', model_location = 'textattack/bert-base-uncased-imdb', detection_method = 'RSV', det_location = 'Detectors/WDR/totalTrain_pwwsVsalbert-base-v2-ag-news_ag-news_train-WDR_RFMODEL.sav', gamma = 0.05, delta = 0.9, delta0_pos = 0, delta1_pos=0, delta_file_loc = 'results/textattack/distilbert-base-uncased-imdb/imdb', delta_weight = 0.5, num_labels = 2):
    obf_method = "GS_MR"
    Detectorattacker = DetectorAttack(obf_method = obf_method, class_alg = classifier, model_location = model_location, detection_method = detection_method, det_location = det_location, gamma = float(gamma), delta = float(delta), delta0_pos = int(delta0_pos), delta1_pos=int(delta1_pos), delta2_pos = 0, delta3_pos=0, delta_file_loc = delta_file_loc, delta_weight = float(delta_weight), num_labels = int(num_labels))
    

    incsv = csv.reader(open(dataset), delimiter = '\t')
    outcsv = csv.writer(open(outfile, 'w'), delimiter = '\t')
  

    num_examples = int(num_examples)
    offset = int(offset)
    if(num_examples == -1):
        num_examples = len(incsv)

    count = 0
    cur_num = 0
    for cur in incsv:
        #head to offset of dataset
        if(not (offset <= cur_num)):
            cur_num += 1
            continue

        if(len(cur) == 2):
            ground_truth = int(cur[1])
            text = cur[0]
        elif(len(cur) == 3): # catches datasets which have idx in first column (e.g. sst2)
            ground_truth = int(cur[2])
            text = cur[1]
        else: # for nli tasks with multiple sentences (assumes idx as well), Detectorattack only attacks the second text. 
            ground_truth = int(cur[3])
            text = (cur[1], cur[2])

        perturbed_text, num_queries = Detectorattacker.obfuscate(text, 0) #reframe attack as attacking not_attack class for detector

        if(type(perturbed_text) == tuple): # for nli tasks
            outcsv.writerow([perturbed_text[0], perturbed_text[1], ground_truth, num_queries])
        else:
            outcsv.writerow([perturbed_text, ground_truth, num_queries])

        count += 1
        if(count >= num_examples):
            break



#RedHerringattacker = RedHerringAttack(obf_method = 'GS_MR', class_alg = 'HuggingFaceModel', model_location = 'textattack/bert-base-uncased-ag-news', detection_method = 'WDR', det_location = 'Detectors/WDR/totalTrain_pwwsVsalbert-base-v2-ag-news_ag-news_train-WDR_RFMODEL.sav')
#out = RedHerringattacker.obfuscate("What's in a Name? Well, Matt Is Sexier Than Paul (Reuters) Reuters - As Shakespeare said, a rose by any other\name would smell as sweet. Right?", 3)
#print(out[0], out[1])


if(__name__ == "__main__"):
    if(len(sys.argv) == 9):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    elif(len(sys.argv) == 11):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10])
    elif(len(sys.argv) == 16):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[10], sys.argv[11], sys.argv[12], sys.argv[13], sys.argv[14], sys.argv[15])


