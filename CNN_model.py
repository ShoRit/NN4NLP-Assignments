import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import random
import time
from torch.utils.data import DataLoader
import sys
import argparse
from sklearn.metrics import *
import os.path
import sklearn

'''
Implementation of the KIM CNN model 

'''

'''
Loading the data from the saved pickle files to avoid the 
hassle of creating them again

'''
with open('/data/rdutt_courses/NN4NLP/word_to_idx.p','rb') as handle:
	word_to_idx= pickle.load(handle)

with open('/data/rdutt_courses/NN4NLP/label_to_idx.p','rb') as handle:
	label_to_idx= pickle.load( handle)

with open('/data/rdutt_courses/NN4NLP/id_to_glove_vec.p','rb') as handle:
	id_to_glove_vec = pickle.load(handle)

with open('/data/rdutt_courses/NN4NLP/id_to_fasttext_vec.p','rb') as handle:
	id_to_fasttext_vec= pickle.load(handle)

with open('/data/rdutt_courses/NN4NLP/id_to_word2vec_vec.p','rb') as handle:
	id_to_word2vec_vec= pickle.load(handle)

with open('/data/rdutt_courses/NN4NLP/train.p','rb') as handle:
	train= pickle.load(handle)

with open('/data/rdutt_courses/NN4NLP/val.p','rb') as handle:
	val= pickle.load(handle)


print("Loading is done")

VOCAB_SIZE= len(word_to_idx)
LABEL_SIZE= len(label_to_idx)
EMB_SIZE= 300
WIN_SIZE=3
FILTER_MAP= 100
BATCH_SIZE= 256
WIN_SIZES=[3,4,5]
DROPOUT=0
LEARNING_RATE= 0.0001
longtype = torch.LongTensor
floattype= torch.FloatTensor
use_cuda = torch.cuda.is_available()


def get_batches(batch):
	label = torch.tensor([entry[1] for entry in batch])
	max_len= max([len(entry[0]) for entry in batch])
	text =[entry[0]+ [word_to_idx['<PAD>'] for i in range(max_len-len(entry[0]))] for entry in batch]
	return text, label

train_data = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
					  collate_fn=get_batches)
valid_data = DataLoader(val, batch_size=32, shuffle=False,
					  collate_fn=get_batches)



class KIM_CNN(torch.nn.Module):
    def __init__(self, nwords, emb_size, num_filters, window_sizes, ntags, trainable_flag, emb_type=None, dropout=0.5):
        super(KIM_CNN, self).__init__()
        
        
        if emb_type=='glove':
            self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_glove_vec))
        elif emb_type=='fasttext':
            self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_fasttext_vec))
        elif emb_type=='word2vec':
        	self.embedding= nn.Embedding.from_pretrained(torch.FloatTensor(id_to_word2vec_vec))
        else:
            self.embedding= nn.Embedding(nwords, emb_size, padding_idx=word_to_idx['<PAD>'])
            nn.init.uniform_(self.embedding.weight, -0.25,0.25)

        self.embedding.weight.requires_grad= trainable_flag
        
        # applying 1 layer convolution 
        
        self.convs= nn.ModuleList([nn.Conv1d(in_channels=emb_size,out_channels=num_filters,kernel_size=window_size) 
                    for window_size in window_sizes])
        
        # self.conv2d= nn.Conv2d(in_channels=emb_size,out_channels=num_filters, kernel_size=window_size)
    
        self.relu= nn.ReLU()
        self.dropout= nn.Dropout(dropout)
        self.projection_layer= nn.Linear(in_features=num_filters*len(window_sizes), out_features=ntags, bias=True)
        nn.init.xavier_uniform_(self.projection_layer.weight)
    
    def forward(self, words, return_activations=False):
        embeds=self.embedding(words) # BATCH_SIZE * n*_words * dim
        embeds=embeds.permute(0,2,1) # BATCH_SIZE * dim *n_words
        h=[conv(embeds) for conv in self.convs] # [BATCH_SIZE * n_filters*(n_words- window_size +1)] * len(WINDOW_SIZE)
        h=[self.relu(h1) for h1 in h]  
        h=[h1.max(dim=2)[0] for h1 in h]
        h= torch.cat(h,1)
        h= self.dropout(h)
        out= self.projection_layer(h)
        return out


def run_model(model):
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	best_dev_acc=0
	best_model=None
	best_epoch=0
	dev_f1=0

	if use_cuda:
		model.cuda()

	for ITER in range(10):
		# Perform training
		# random.shuffle(train)
		train_loss = 0.0
		train_accs=[]
		train_f1s=[]
		val_accs=[]
		val_f1s=[]
		start = time.time()
		valid_loss=0.0
		valid_correct=0


		for index, (text,labels) in enumerate(train_data):
			text= torch.tensor(text).type(longtype)
			labels= labels.type(longtype)
			scores=model(text)
			predict=[score.argmax().item() for score in scores]

			train_accs.append(sklearn.metrics.accuracy_score(labels,predict))
			train_f1s.append(sklearn.metrics.f1_score(labels, predict, average='macro'))
			my_loss = criterion(scores, labels)
			train_loss += my_loss.item()
			optimizer.zero_grad()
			my_loss.backward()
			optimizer.step()

		print("iter %r: train loss/sent=%.4f, acc=%.4f, f1=%.4f, time=%.2fs" % (
			ITER, train_loss / len(train), np.mean(train_accs), np.mean(train_f1s), time.time() - start))

		for index, (text,labels) in enumerate(valid_data):
			text= torch.tensor(text).type(longtype)
			labels= labels.type(longtype)
			scores=model(text)
			predict=[score.argmax().item() for score in scores]
			val_accs.append(sklearn.metrics.accuracy_score(labels, predict))
			val_f1s.append(sklearn.metrics.f1_score(labels,predict, average= 'macro'))

		if np.mean(val_accs) > best_dev_acc:
			# best_model=model
			best_dev_acc= np.mean(val_accs)
			dev_f1= np.mean(val_f1s)
			best_epoch= ITER

		# if np.mean(val_f1s)> best_dev_f1:
		# 	best_dev_f1= np.mean(val_f1s)


		print("iter %r: test acc=%.4f, test f1=%.4f," % (ITER, np.mean(val_accs), np.mean(val_f1s)))
	print('Best dev accuracy is ='+ str(round(best_dev_acc,4)) + ' at epoch = '+ str(best_epoch)) 
	print('Corresponding F1 accuracy_score is ='+ str(round(dev_f1,4)) + ' at epoch = '+ str(best_epoch))

	return best_dev_acc, dev_f1


model_dict={}
# model_dict={}
# if os.path.isfile('model_dict.p'):
# 	with open('model_dict.p','rb') as handle:
# 		model_dict=pickle.load(handle)

emb_type=None
trainable_flag=False
emb_train='dynamic'

try:
	emb_type= sys.argv[1]
	if emb_type not in ['fasttext','glove','word2vec']:
		emb_type='random'
except Exception as e:
	emb_type='random'

try:
	emb_train=sys.argv[2]
	if emb_train=='dynamic':
		trainable_flag=True
except Exception as e:
	trainable_flag=False

try:
	gpu_device= int(sys.argv[3])
except Exception as e:
	gpu_device=0

try:
	DROPOUT= float(sys.argv[4])
except Exception as e:
	DROPOUT= 0

try:
	BATCH_SIZE= int(sys.argv[5])
except Exception as e:
	BATCH_SIZE= 256

if use_cuda:
	longtype = torch.cuda.LongTensor
	floattype= torch.cuda.FloatTensor
	torch.cuda.set_device(gpu_device)


model_name= 'KIM_CNN'+'_'+emb_type+'_'+emb_train+'_'+str(DROPOUT)+'_'+str(BATCH_SIZE)
model_dict[model_name]={}
model_dict[model_name]['acc']=[]
model_dict[model_name]['f1']=[]

for i in range(5):
	model = KIM_CNN(VOCAB_SIZE, EMB_SIZE, FILTER_MAP, WIN_SIZES, LABEL_SIZE, 
	trainable_flag, emb_type ,DROPOUT)
	acc,f1= run_model(model)
	model_dict[model_name]['acc'].append(acc)
	model_dict[model_name]['f1'].append(f1)

with open(model_name+'.p','wb') as handle:
	pickle.dump(model_dict, handle)


