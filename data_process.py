'''
Basic file to process and load data for the models.

Authors : Ritam Dutt (rdutt)
'''

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

only_letters= r'[^A-Za-z0-9\s]+'

def text_preprocess(sent):
	sent=sent.lower().strip()
	sent=re.sub(only_letters,'',sent)
	sent=re.sub("\s\s+" , " ", sent)
	return sent

data_dir='topicclass/'
train_file='topicclass_train.txt'
val_file='topicclass_valid.txt'
test_file='topicclass_test.txt'

'''
Create the train, validation and test dataframes

'''

def create_data_frame(file_name):
	fp= open(file_name)
	data={'Sentence':[], 'Label':[]}
	for line in fp:
		try:
			line=line.strip().split('|||')
			sent= text_preprocess(line[1]).strip()
			label=line[0].strip()
			data['Sentence'].append(sent)
			data['Label'].append(label)
		except Exception as e:
			print(e)
	
	df= pd.DataFrame(data)
	return df

train_df= create_data_frame(data_dir+train_file)
val_df= create_data_frame(data_dir+val_file)
# test_df= create_data_frame(data_dir+ test_file)

print("Total number of training sentences = ", len(train_df))
print("Total number of validation sentences = ",len(val_df))

'''
Create ids for the words and labels only on the TRAINING DATA. Thus 
any new word in the validation data would be generalized to <UNK>. 
'''

word_to_idx={}
label_to_idx={}
word_to_idx['unk']=0
word_to_idx['<PAD>']=1
word_count=len(word_to_idx)

train_sents= list(train_df['Sentence'])
for sent in train_sents:
	words= sent.split()
	for word in words:
		if word not in word_to_idx:
			word_to_idx[word]=word_count
			word_count+=1


print("Vocabulary size = ", len(word_to_idx))

labels= set(train_df['Label'])
for label in labels:
	label_to_idx[label]=len(label_to_idx)
print("Number of labels =", len(label_to_idx))


'''
Functions to create vector embeddings for the words in the text
In this analysis, we focus on the following 3 vector embeddings

- word2vec
- GloVe
- fasttext

'''

def create_glove_vectors(glove_data_file='/data/glove_vector/glove.6B.300d.txt'):
	glove_dict={}
	data_file= open(glove_data_file)
	for line in data_file:
		line=line.strip().split()
		glove_dict[line[0]]=np.float32(np.array([float(i) for i in line[1:]]))
	return glove_dict
glove_dict= create_glove_vectors()

def create_fasttext_vectors(fasttext_data_file='/data/fasttext_vectors/crawl-300d-2M-subword.vec'):
	fasttext_dict={}
	data_file= open(fasttext_data_file,encoding='utf-8', newline='\n', errors='ignore')
	for line in data_file:
		line=line.strip().split()
		if len(line)==2:
			continue
		fasttext_dict[line[0]]=np.float32(np.array([float(i) for i in line[1:]]))
	return fasttext_dict
fasttext_dict=create_fasttext_vectors()

def create_word2vec_vectors(file='/data/GoogleNews-vectors-negative300.txt'):
	word2vec_dict={}
	data_file=open(file)
	for line in data_file:
		line=line.strip().split()
		if len(line)==2:
			continue
		fasttext_dict[line[0]]=np.float32(np.array([float(i) for i in line[1:]]))
	return fasttext_dict

word2vec_dict= create_word2vec_vectors()

'''
create the vectors for each sentence/ text instance for the 
train, validation and the test set
'''

def create_sent_vectors(df, test_flag=False):
	data=[]
	for index, row in df.iterrows():
		words=[]
		words.extend([word_to_idx[word] if word in word_to_idx else word_to_idx['unk'] for word in row['Sentence'].split()])
		if test_flag:
			label='<TBD>'
		else:
			label=label_to_idx[row['Label']]   
		data.append((words,label))

	return data

train= create_sent_vectors(train_df)
val= create_sent_vectors(val_df)
# test= create_sent_vectors(test_df, test_flag=True)   


'''
Creates weight matrices for the different kinds of embeddings so
that they can be loaded into Pytorch

'''

def create_weight_matrix(word_to_idx, glove_dict, fasttext_dict, 
	embed_dim=300):
	id_to_word2vec_vec= np.zeros((len(word_to_idx), embed_dim))
	id_to_glove_vec=np.zeros((len(word_to_idx), embed_dim ))
	id_to_fasttext_vec=np.zeros((len(word_to_idx), embed_dim))
	
	for w in word_to_idx:
		if w=='<PAD>':
			continue
		if w in glove_dict:
			id_to_glove_vec[word_to_idx[w]]= glove_dict[w]
		else:
			id_to_glove_vec[word_to_idx[w]]= glove_dict['unk']
	
		if w in fasttext_dict:
			id_to_fasttext_vec[word_to_idx[w]]= fasttext_dict[w]
		else:
			id_to_fasttext_vec[word_to_idx[w]]= fasttext_dict['unk']
	
		if w in word2vec_dict:
			id_to_word2vec_vec[word_to_idx[w]]= word2vec_dict[w]
		else:
			id_to_word2vec_vec[word_to_idx[w]]= word2vec_dict['unk']

	return id_to_glove_vec, id_to_fasttext_vec, id_to_word2vec_vec

id_to_glove_vec, id_to_fasttext_vec, id_to_word2vec_vec = create_weight_matrix(word_to_idx, glove_dict,
 fasttext_dict, 300)


'''
Dumps the vectors in pickle file for faster loading. 
'''


with open('/data/rdutt_courses/NN4NLP/word_to_idx.p','wb') as handle:
	pickle.dump(word_to_idx, handle)

with open('/data/rdutt_courses/NN4NLP/label_to_idx.p','wb') as handle:
	pickle.dump(label_to_idx, handle)

with open('/data/rdutt_courses/NN4NLP/word2vec_dict.p','wb') as handle:
	pickle.dump(word2vec_dict, handle)

with open('/data/rdutt_courses/NN4NLP/fasttext_dict.p','wb') as handle:
	pickle.dump(fasttext_dict, handle)

with open('/data/rdutt_courses/NN4NLP/glove_dict.p','wb') as handle:
	pickle.dump(glove_dict, handle)

with open('/data/rdutt_courses/NN4NLP/id_to_glove_vec.p','wb') as handle:
	pickle.dump(id_to_glove_vec,handle)

with open('/data/rdutt_courses/NN4NLP/id_to_fasttext_vec.p','wb') as handle:
	pickle.dump(id_to_fasttext_vec,handle)

with open('/data/rdutt_courses/NN4NLP/id_to_word2vec_vec.p','wb') as handle:
	pickle.dump(id_to_word2vec_vec,handle)

with open('/data/rdutt_courses/NN4NLP/train.p','wb') as handle:
	pickle.dump(train, handle)

with open('/data/rdutt_courses/NN4NLP/val.p','wb') as handle:
	pickle.dump(val,handle)
