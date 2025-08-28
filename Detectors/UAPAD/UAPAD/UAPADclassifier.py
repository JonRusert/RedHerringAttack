# functions to use UAPAD as detector for redherringattack
# Example:

import sys
import os
import csv
import json
import transformers
sys.path.append("..")
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, T5ForConditionalGeneration
)


class UAPADDetector:


    def __init__(self, hfmodel = 'textattack/distilbert-base-uncased-imdb', delta0_pos = 0, delta1_pos = 0, delta2_pos = 0, delta3_pos = 0, delta_file_loc = 'results/textattack/distilbert-base-uncased-imdb/imdb', delta_weight = 0.5, num_labels = 2):
        # pre-trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config, self.tokenizer, self.model = load_pretrained_models(hfmodel, num_labels)
        self.model.to(self.device)

        # adv + delta feature
        with open(os.path.join(delta_file_loc, 'deltas0.json'), 'r') as f1:
            self.delta0s = json.load(f1)
        with open(os.path.join(delta_file_loc, 'deltas1.json'), 'r') as File2:
            self.delta1s = json.load(File2)
            
        if(num_labels == 4):
            with open(os.path.join(delta_file_loc, 'deltas2.json'), 'r') as f3:
                self.delta2s = json.load(f3)
            with open(os.path.join(delta_file_loc, 'deltas3.json'), 'r') as File4:
                self.delta3s = json.load(File4)
        
        self.delta0_pos = delta0_pos
        self.delta1_pos = delta1_pos
        self.delta2_pos = delta2_pos
        self.delta3_pos = delta3_pos
        self.delta_weight = delta_weight
        self.num_labels = num_labels



    def predict(self, test_query, attacked_class = 0):
        model_inputs = self.tokenizer([test_query],  return_tensors='pt', truncation=True, padding=True)
        #if(len(test_query.split()) > 512):
        #    return 0, 1.0

        embeddings0 = []
        embeddings1 = []
        embeddings2 = []
        embeddings3 = []
        label_list = []

        
        self.model.zero_grad()
        word_embedding_layer = self.model.get_input_embeddings()
        input_ids = model_inputs['input_ids']
        del model_inputs['input_ids']  # new modified
        attention_mask = model_inputs['attention_mask']
        embedding_init = word_embedding_layer(input_ids)

        input_mask = attention_mask.to(embedding_init)
        input_lengths = torch.sum(input_mask, 1)
        mask_tensor = input_mask.unsqueeze(2)
        repeat_shape = mask_tensor.shape    

        delta0 = torch.tensor(self.delta0s[self.delta0_pos]).to(self.device)
        delta1 = torch.tensor(self.delta1s[self.delta1_pos]).to(self.device)
        if self.num_labels == 4:
            delta2 = torch.tensor(self.delta2s[self.delta2_pos]).to(self.device)
            delta3 = torch.tensor(self.delta3s[self.delta3_pos]).to(self.device)


        model_inputs['inputs_embeds'] = embedding_init
        if(isinstance(self.model, transformers.T5ForConditionalGeneration)):
            logits = model(**model_inputs, labels = input_ids).logits
            probabilities = torch.softmax(logits, dim=-1)
            generated_tokens = torch.argmax(logits, dim=-1)
            tok_probs = []
            for j in range(len(generated_tokens)):
                    gens = generated_tokens[j]
                    cur_probs = probabilities[j]
                    generated_text = tokenizer.decode(gens)
                    #print(generated_text)
                    tok_probs.append(get_token_probs(tokenizer, gens, cur_probs))
            logits = torch.tensor(tok_probs)
            logits_check = logits.to(device)
        else:
            logits_check = self.model(**model_inputs).logits
        _, preds_check = logits_check.max(dim=-1)
            
        batch_delta = torch.zeros(embedding_init.size()).to(self.device)

        for i in range(preds_check.size()[0]):
            if preds_check[i] == 1:
                batch_delta[i,:,:] = -delta1.repeat(repeat_shape[1], 1)
                # batch_delta[i, :, :] = delta0.repeat(repeat_shape[1], 1)
            elif preds_check[i] == 0:
                batch_delta[i,:,:] = -delta0.repeat(repeat_shape[1], 1)
                # batch_delta[i, :, :] = delta1.repeat(repeat_shape[1], 1)
            elif preds_check[i] == 2:
                batch_delta[i,:,:] = -delta2.repeat(repeat_shape[1], 1)
            elif preds_check[i] == 3:
                batch_delta[i,:,:] = -delta3.repeat(repeat_shape[1], 1)
            else:
                assert ValueError

        

        #pp = embedding_init * mask_tensor
        #for i in range(pp.shape[0]):
        #    embd0 = torch.matmul(pp[i,:,:].squeeze(0), delta0) / torch.norm(delta0)
        #    embd1 = torch.matmul(pp[i,:,:].squeeze(0), delta1) / torch.norm(delta1)
        #    embeddings0.append(embd0)
        #    embeddings1.append(embd1)
        #    label_list.append(labels[i])


        model_inputs['inputs_embeds'] = embedding_init + (self.delta_weight*batch_delta*mask_tensor).to(torch.float32)

        if(isinstance(self.model, transformers.T5ForConditionalGeneration)):
            logits = model(**model_inputs, labels = input_ids).logits
            probabilities = torch.softmax(logits, dim=-1)
            generated_tokens = torch.argmax(logits, dim=-1)
            tok_probs = []
            for j in range(len(generated_tokens)):
                    gens = generated_tokens[j]
                    cur_probs = probabilities[j]
                    generated_text = tokenizer.decode(gens)
                    #print(generated_text)
                    tok_probs.append(get_token_probs(tokenizer, gens, cur_probs))
            logits = torch.tensor(tok_probs)
            logits = logits.to(device)
            _, preds = logits.max(dim=-1)
            predict_logits = logits.detach().cpu()
            check_logits = logits_check.detach().cpu()
        else:
            logits = self.model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            predict_logits = torch.softmax(logits, dim=1).detach().cpu()
            check_logits = torch.softmax(logits_check, dim=1).detach().cpu()
        
        pred_labels = preds.detach().cpu().numpy()
        check_labels = preds_check.detach().cpu().numpy()
        #true_labels = labels.detach().cpu().numpy()
        
    
        adv_pred_labels = np.array([
        0 if pred_labels[i] == check_labels[i] else 1 for i in range(len(check_labels))
        ])

        return adv_pred_labels[0], predict_logits[0][attacked_class]

        


    # allow multiple queries in form of list
    def predictMultiple(self, test_queries, attacked_class = 0):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test, attacked_class)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs



# modified from adv_detect.py
def load_pretrained_models(model_name, num_labels):
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels, max_length=512)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    
    if('t5' in model_name):
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    
    return config, tokenizer, model










#detector = UAPADDetector(hfmodel = 'textattack/distilbert-base-uncased-imdb', delta_file_loc = 'results/textattack/distilbert-base-uncased-imdb/imdb', delta_weight = 0.4, num_labels = 2)

#print(detector.predict("This movie is the worst movie ever!!!"))
