# functions to train, test,and store classification model for adaboost for WDR data.
# Example:

import sys
import csv
sys.path.append('../../Classifiers/GooglePerspective/')
sys.path.append('../../Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
import pickle
import xgboost as xgb



class WDRDetector:
    
    def __init__(self, det_location = 'totalTrain_pwwsVsalbert-base-v2-ag-news_ag-news_train-WDR_RFMODEL.sav', class_alg = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-ag-news'):
        self.detector = pickle.load(open(det_location, 'rb'))
        self.max_len = self.detector.n_features_in_

        if(class_alg == 'Perspective'):
            self.classifier = Perspective(threshold = 0.5, select = persp_key)
        elif(class_alg == 'HuggingFaceModel'):
            self.classifier = HuggingFaceModel(hfmodel)


    def predict(self, test_query, attacked_class = 0):
        beg_sentence = None

        orig_query = test_query
        if(type(test_query) == tuple): # need to get WDR of only second sentence
            beg_sentence = test_query[0]
            test_query = test_query[1]

        #calculate WDR for test query to make prediction
        split_text = test_query.split()
        variations = []

        #calculate list of all words missing from text
        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_text)):
            modified_query = ' '.join(split_text[:cur_pos] + ['[UNK]'] + split_text[cur_pos+1:])
            
            if(beg_sentence): #nli tasks, attack second sentence, but need first for classification
                variations.append((beg_sentence, modified_query))
            else:
                variations.append(modified_query)

        #calculate wdrs
        # calculate original for use in WDR calculations
        orig_pred, orig_logits = self.classifier.logits(orig_query)
        orig_pred = int(orig_pred)
        orig_pred_logit = orig_logits[orig_pred]
        orig_logits.pop(orig_pred)
        second_logit = max(orig_logits)
        orig_wdr = orig_pred_logit - second_logit

        wdrs = [orig_wdr]
        for cur in variations:

            # get logits
            cur_pred, cur_logits = self.classifier.logits(cur)
            #print(cur_pred, cur_logits)
            #cur_pred = int(cur_pred)

            # calculate WDR
            #pred_logit = cur_logits[cur_pred]
            #cur_logits.pop(cur_pred)
            pred_logit = cur_logits[orig_pred]
            cur_logits.pop(orig_pred)

            second_logit = max(cur_logits)

            wdr = pred_logit - second_logit


            wdrs.append(wdr)

            
        wdrs.sort(reverse=True, key = abs)

        #pad or shorten test data to self.max_len
        while(len(wdrs) < self.max_len):
            wdrs.append(0.0)
            
        while(len(wdrs) > self.max_len):
            wdrs.pop(len(wdrs) - 1)
        
        
        # detector makes prediction if attack is occuring or not
        pred = self.detector.predict([wdrs])[0]
        prob = self.detector.predict_proba([wdrs])[0][attacked_class]
        
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



def trainTestWDR(trainfile, testfile):
    # process training/test data
    traincsv = csv.reader(open(trainfile, 'r'), delimiter ='\t')
    testcsv = csv.reader(open(testfile, 'r'), delimiter = '\t')

    train_x = []
    train_y = []
    test_x =[]
    test_y = []

    # load in WDR scores, sort and pad to max length of train data
    max_len = 512
    for cur in traincsv:
        cur_label = int(cur[-1])
        cur_data = [float(x) for x in cur[:-1]]
        cur_data.sort(reverse=True, key = abs)
        
        # trim if greater than max_len
        if(len(cur_data) > max_len):
            cur_data = cur_data[:max_len]

        train_x.append(cur_data)
        train_y.append(cur_label)

        #if(len(cur_data) > max_len):
        #    max_len = len(cur_data)
        

    #print(max_len)
    # pad train data
    for i in range(len(train_x)):
        while(len(train_x[i]) < max_len):
            train_x[i].append(0.0)
            
    # load in test and pad
    for cur in testcsv:
        cur_label = int(cur[-1])
        cur_data = [float(x) for x in cur[:-1]]
        cur_data.sort(reverse=True, key = abs)
        
        #pad or shorten test data to max_len
        while(len(cur_data) < max_len):
            cur_data.append(0.0)
            
        while(len(cur_data) > max_len):
            cur_data.pop(len(cur_data) - 1)
            
        test_x.append(cur_data)
        test_y.append(cur_label)
        
    train_x, train_y = shuffle(train_x, train_y)
    adversarial_x = [test_x[i] for i in range(len(test_x)) if test_y[i] == 1]
    adv_test_x = [1] * len(adversarial_x)

    #print(type(train_x))
    #print(type(train_y))
    #print([type(t) for t in train_x])
    # train adaboost
    clf = AdaBoostClassifier(random_state = 0)
    clf.fit(train_x, train_y)
    
    train_score = clf.score(train_x, train_y)
    test_score = clf.score(test_x, test_y)

    preds = clf.predict(test_x)
    f1 = f1_score(preds, test_y)
    
    adv_preds = clf.predict(adversarial_x)
    adv_recall = recall_score(adv_test_x, adv_preds)
    
    print("------ADABoost-------")
    print("Train score:", train_score)
    print("Test score", test_score)
    print("F1 - test:", f1)
    print("Adv. Recall:", adv_recall)
    
    clf = RandomForestClassifier(n_estimators=1600,
                                 min_samples_split=10,
                                 min_samples_leaf=2,
                                 max_features='auto',
                                 max_depth=None, 
                                 bootstrap = True, 
                                 random_state = 0)
    clf.fit(train_x, train_y)
    
    train_score = clf.score(train_x, train_y)
    test_score = clf.score(test_x, test_y)
    preds = clf.predict(test_x)
    f1 = f1_score(preds, test_y)
    adv_preds = clf.predict(adversarial_x)
    adv_recall = recall_score(adv_test_x, adv_preds)

    
    filename = trainfile.split('.')[0] + '_RFMODEL.sav'
    pickle.dump(clf, open(filename, 'wb'))

    print("------RF-------")
    print("Train score:", train_score)
    print("Test score", test_score)
    print("F1 - test:", f1)
    print("Adv. Recall:", adv_recall)


    
    clf = MLPClassifier(random_state = 0)
    clf.fit(train_x, train_y)
    
    train_score = clf.score(train_x, train_y)
    test_score = clf.score(test_x, test_y)
    preds = clf.predict(test_x)
    f1 = f1_score(preds, test_y)
    adv_preds = clf.predict(adversarial_x)
    adv_recall = recall_score(adv_test_x, adv_preds)
    
    print("------MLP-------")
    print("Train score:", train_score)
    print("Test score", test_score)
    print("F1 - test:", f1)
    print("Adv. Recall:", adv_recall)


    #clf = pickle.load(open('wdr/Classifier/ag-news_classifier.pickle', 'rb'))
    xgb_classifier = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.34281802,
                    gamma=0.6770816,
                    min_child_weight=2.5520658,
                    max_delta_step=0.71469694,
                    subsample=0.61460966,
                    colsample_bytree=0.73929816,
                    colsample_bylevel=0.87191725,
                    reg_alpha=0.9064181,
                    reg_lambda=0.5686102,
                    n_estimators=29,
                    silent=0,
                    nthread=4,
                    scale_pos_weight=1.0,
                    base_score=0.5,
                    missing=None,
                  )

    xgb_classifier.fit(train_x, train_y)
    train_score = xgb_classifier.score(train_x, train_y)
    test_score = xgb_classifier.score(test_x, test_y)
    preds = xgb_classifier.predict(test_x)
    f1 = f1_score(preds, test_y)
    adv_preds = xgb_classifier.predict(adversarial_x)
    adv_recall = recall_score(adv_test_x, adv_preds)

    filename = trainfile.split('.')[0] + '_XGBMODEL.sav'
    pickle.dump(clf, open(filename, 'wb'))


    print("------XGBoost-------")
    print("Train score:", train_score)
    print("Test score", test_score)
    print("F1 - test:", f1)
    print("Adv. Recall:", adv_recall)
    print(classification_report(test_y, preds, digits=3))


if(__name__ == "__main__"):
    trainTestWDR(sys.argv[1], sys.argv[2])
