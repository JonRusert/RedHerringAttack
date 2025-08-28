import sys
import csv
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration


class HuggingFaceModel:
    
    def __init__(self, hfmodel = 'textattack/bert-base-uncased-imdb'):
        self.tokenizer = AutoTokenizer.from_pretrained(hfmodel)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if('t5' in hfmodel):
            self.t5 = True
            self.model = T5ForConditionalGeneration.from_pretrained(hfmodel)
        else:
            self.t5 = False
            self.model = AutoModelForSequenceClassification.from_pretrained(hfmodel)


    def predict(self, test_query, attacked_class = 0):
        if(self.t5):
            return self.gen_predict(test_query, attacked_class) #added to support LLM classification
        else:
            return self.class_predict(test_query, attacked_class)

    def gen_predict(self, test_query, attacked_class):
        text_input_list = [test_query]
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            #padding="max_length",
            #max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = self.tokenizer(test_query, return_tensors='pt')['input_ids']
        #print(input_ids)
        logits = self.model(**inputs_dict, labels = input_ids).logits
        probabilities = torch.softmax(logits, dim=-1)
        generated_tokens = torch.argmax(logits, dim=-1)
        #generated_text = self.tokenizer.decode(generated_tokens[0])
        #print(probabilities)
        #token_id_0 = self.tokenizer.encode('0', add_special_tokens=False)[1]
        #token_id_1 = self.tokenizer.encode('1', add_special_tokens=False)[0] # only supporting binary pred, need to add more class if beyond
        token_id_0 = self.tokenizer.encode('negative', add_special_tokens=False)[0]
        token_id_1 = self.tokenizer.encode('positive', add_special_tokens=False)[0]
        #print(self.tokenizer.encode('1', add_special_tokens=False))
        #print(token_id_0)
        #print(token_id_1)
        token_position = 0
        for i in range(len(generated_tokens[0])):
            if(generated_tokens[0][i] == token_id_0 or generated_tokens[0][i] == token_id_1):
                token_position = i
                break
                

        #token_id = generated_tokens[0, token_position].item()
        token_probability_0 = probabilities[0, token_position, token_id_0].item()
        token_probability_1 = probabilities[0, token_position, token_id_1].item()
        #print(token_position)
        if(token_probability_0 > token_probability_1):
            pred = 0
        else:  
            pred = 1

        probs = [token_probability_0, token_probability_1]
        
        return pred, probs[attacked_class]
        



    def class_predict(self, test_query, attacked_class = 0):
        #print(test_query)
        
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        
        text_input_list = [test_query]

        self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        

        #tok_text = self.tokenizer([test_query], return_tensors='pt')
        #print(tok_text)
        #output = self.model(**tok_text)[0]
        output = self.model(**inputs_dict)[0]

        #print(output)
        softmax = torch.nn.Softmax(dim=1)
        output_sm = softmax(output)[0]
        #print(output_sm)
            
        pred = float(output_sm.argmax())
        
        prob = output_sm[int(attacked_class)].detach().numpy()
        
        return pred, float(prob)



    # allow multiple queries in form of list
    def predictMultiple(self, test_queries, attacked_class = 0):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test, attacked_class)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs



    def logits(self, test_query):
        if(self.t5):
            return self.gen_logits(test_query) #added to support LLM classification
        else:
            return self.class_logits(test_query)

        
    #returns prediction logits for all classes
    def class_logits(self, test_query):
        #print(test_query)
        
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        
        text_input_list = [test_query]

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        

        #tok_text = self.tokenizer([test_query], return_tensors='pt')
        #print(tok_text)
        #output = self.model(**tok_text)[0]
        output = self.model(**inputs_dict)[0]

        pred = float(output.argmax())
            
        output = output.detach().numpy()[0].tolist()
        
        return pred, output


    def gen_logits(self, test_query):
        text_input_list = [test_query]
        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            #padding="max_length",
            #max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = self.tokenizer(test_query, return_tensors='pt')['input_ids']
        #print(input_ids)
        logits = self.model(**inputs_dict, labels = input_ids).logits
        #probabilities = torch.softmax(logits, dim=-1)
        generated_tokens = torch.argmax(logits, dim=-1)
        #generated_text = self.tokenizer.decode(generated_tokens[0])
        #print(probabilities)
        #token_id_0 = self.tokenizer.encode('0', add_special_tokens=False)[1]
        #token_id_1 = self.tokenizer.encode('1', add_special_tokens=False)[0] # only supporting binary pred, need to add more class if beyond
        token_id_0 = self.tokenizer.encode('negative', add_special_tokens=False)[0]
        token_id_1 = self.tokenizer.encode('positive', add_special_tokens=False)[0]
        #print(self.tokenizer.encode('1', add_special_tokens=False))
        #print(token_id_0)
        #print(token_id_1)
        token_position = 0
        for i in range(len(generated_tokens[0])):
            if(generated_tokens[0][i] == token_id_0 or generated_tokens[0][i] == token_id_1):
                token_position = i
                break
                

        #token_id = generated_tokens[0, token_position].item()
        token_probability_0 = logits[0, token_position, token_id_0].item()
        token_probability_1 = logits[0, token_position, token_id_1].item()
        #print(token_position)
        if(token_probability_0 > token_probability_1):
            pred = 0
        else:  
            pred = 1

        probs = [token_probability_0, token_probability_1]
        
        return pred, probs
    

    # allow multiple queries in form of list
    def logitsMultiple(self, test_queries):
        logits = []
        preds = []
        for test in test_queries:
            cur_pred, cur_logits = self.logits(test)
            logits.append(cur_logits)
            preds.append(cur_pred)

        return preds, logits






'''
text= 'this movie was very good 10/10'

tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')

tok_text = tokenizer([text], return_tensors='pt')
print(tok_text)
output = model(**tok_text)[0]
print(output)
softmax = torch.nn.Softmax(dim=1)
output_sm = softmax(output)[0]

pred = float(output_sm.argmax())
        
        
print(output_sm)
print(pred)


hfm = HuggingFaceModel()
pos_text = 'this movie was very good 10/10'
neg_text = 'this movie was horrible 0/10'
print(hfm.predict(pos_text, 1))
print(hfm.predict(neg_text, 0))
'''

