import pandas as pd
import numpy as np
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

## Plot
import plotly
import plotly.graph_objs as go
import matplotlib as plt

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

subreddits = ['reddit-normal', 'depression-sample']
subreddits.sort()
def main(ngram_range=(1,2)):
	#import ml
	import numpy
	import itertools
	import pickle
	import os
	import content
	import classification
	fname = "combinations-10fold.pickle"
	combinations = list(itertools.combinations(subreddits, 2))
	print (combinations)
	for combination in combinations:
		
		if os.path.isfile(fname):
			f = open(fname, "r")
			rs = pickle.load(f)
			f.close()
		else:
			rs = {}

		if combination[0] in rs.keys():
			if combination[1] in rs[combination[0]].keys():
				print (combination, "already exists, skipping...")
				continue

		print ("doing", combination[0], "-", combination[1])
		df1 = pd.read_pickle(combination[0] + ".pickle")
		#by default index will be your row number
		#changes random index numbers to start from 0
		df1 = df1.reset_index()
		print (df1.columns)
		#applies the getTextFromRecord for each row of df since axis=1
		df1['text'] = df1.apply(content.getTextFromRecord, axis=1)
		#changes the index ordering to some random list of row numbers
		df1 = df1.reindex(numpy.random.permutation(df1.index))
		df2 = pd.read_pickle(combination[1] + ".pickle")
		df2 = df2.reset_index()
		df2['text'] = df2.apply(content.getTextFromRecord, axis=1)
		df2 = df2.reindex(numpy.random.permutation(df2.index))
		# keep only posts, keep only the text column
		df1 = df1[df1['parent_id'].astype(str)!='nan']
		#if any of the text column is NaN then drop such rows
		df1 = df1.dropna(subset=['text'])		
		df2 = df2[df2['parent_id'].astype(str)!='nan']
		df2 = df2.dropna(subset=['text'])
		print ("After preprocessing the length of the Dataframe1:"+str(len(df1))+" length of Dataframe2:"+str(len(df2)))
		results = []
		
		print ("choosing min from", len(df1), len(df2))
		m = min(len(df1), len(df2))
		#for i in range(0,10):
		df1_min = df1.reindex(numpy.random.permutation(df1.index)).head(m)	
		df2_min = df2.reindex(numpy.random.permutation(df2.index)).head(m)	
		#extract only the text part of the dataframe with classification as class1-for depression and class2-for suicide
		import Cnn
		data = Cnn.getData(df1_min, df2_min)
		#data['text']=data['text'].map(lambda x: Cnn.clean_text(x))
		vocabulary_size = 20000
		tokenizer = Tokenizer(num_words= vocabulary_size)
		tokenizer.fit_on_texts(data['text'])
		sequences = tokenizer.texts_to_sequences(data['text'])
		datax = pad_sequences(sequences, maxlen=50)
		print(datax.shape)
		df_x=datax
		df_y=np.array(data['class'])
		x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


		def create_conv_model():
			model_conv = Sequential()
			model_conv.add(Embedding(vocabulary_size, 100, input_length=50))
			model_conv.add(Dropout(0.2))
			model_conv.add(Conv1D(64, 5, activation='relu'))
			model_conv.add(MaxPooling1D(pool_size=4))
			model_conv.add(LSTM(100))
			model_conv.add(Dense(1, activation='sigmoid'))
			model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
			return model_conv
		model_conv = create_conv_model()
		history=model_conv.fit(x_train, y_train,epochs=50, batch_size=len(datax), verbose=2)
		print "CNN+Word2Vec"
		y_pred = model_conv.predict(x_test)
		y_pred=(y_pred>0.5)
		tn,fp,fn,tp=confusion_matrix(y_test, y_pred).ravel()
    	f1score=f1_score(y_test, y_pred)
    	recall=recall_score(y_test, y_pred)
    	precision=precision_score(y_test, y_pred)
    	print "recall_score",recall
    	print "precision_score",precision
    	print "ERDE score"
    	erdeFP=tp*fp/(tp+tn+fp+fn)
    	erdeFN=fn
    	epowerval=1/(1+math.exp(3-5)) #3-no of test split #5-ERDE5 standard
    	erdeTP=(1-epowerval)*tp
    	erdeTN=0
    	print "*****ERDE accuracy score*******"
    	print "ERDEtp",erdeTP
    	print "ERDEtn",erdeTN
    	print "ERDEfp",erdeFP
    	print "ERDEfn",erdeFN
    	ERDEscore=(erdeTP+erdeTN)/(erdeTP+erdeTN+erdeFP+erdeFN)
    	print ERDEscore
	embeddings_index = dict()
	f = open('glove.6B/glove.6B.100d.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	print('Loaded %s word vectors.' % len(embeddings_index))
	# create a weight matrix for words in training docs
	embedding_matrix = np.zeros((vocabulary_size, 100))
	for word, index in tokenizer.word_index.items():
		if index > vocabulary_size - 1:
			break
		else:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector
	print "over"
	model_glove = Sequential()
	model_glove.add(Embedding(vocabulary_size, 100, input_length=50, weights=[embedding_matrix], trainable=False))
	model_glove.add(Dropout(0.2))
	model_glove.add(Conv1D(64, 5, activation='relu'))
	model_glove.add(MaxPooling1D(pool_size=4))
	model_glove.add(LSTM(100))
	model_glove.add(Dense(1, activation='sigmoid'))
	model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	history=model_glove.fit(x_train, y_train,epochs=50, batch_size=len(datax), verbose=2)
	print "CNN"
	y_pred = model_glove.predict(x_test)
	y_pred=(y_pred>0.5)
	tn,fp,fn,tp=confusion_matrix(y_test, y_pred).ravel()
	f1score=f1_score(y_test, y_pred)
	recall=recall_score(y_test, y_pred)
	precision=precision_score(y_test, y_pred)
	print "recall_score",recall
	print "precision_score",precision
	print "ERDE score"
	erdeFP=tp*fp/(tp+tn+fp+fn)
	erdeFN=fn
	epowerval=1/(1+math.exp(3-5)) #3-no of test split #5-ERDE5 standard
	erdeTP=(1-epowerval)*tp
	erdeTN=0
	print "*****ERDE accuracy score*******"
	print "ERDEtp",erdeTP
	print "ERDEtn",erdeTN
	print "ERDEfp",erdeFP
	print "ERDEfn",erdeFN
	ERDEscore=(erdeTP+erdeTN)/(erdeTP+erdeTN+erdeFP+erdeFN)
	print ERDEscore

	# conv_embds = model_conv.layers[0].get_weights()[0]
	# glove_emds = model_glove.layers[0].get_weights()[0]
	# word_list = []
	# for word, i in tokenizer.word_index.items():
	# 	word_list.append(word)
	# def plot_words(data, start, stop, step):
	# 	trace = go.Scatter(
	# 		x = data[start:stop:step,0],
	# 		y = data[start:stop:step, 1],
	# 		mode = 'markers',
	# 		text= word_list[start:stop:step]
	# 	)
	# 	layout = dict(title= 't-SNE 1 vs t-SNE 2',
	# 				yaxis = dict(title='t-SNE 2'),
	# 				xaxis = dict(title='t-SNE 1'),
	# 				hovermode= 'closest')
	# 	fig = dict(data = [trace], layout= layout)
	# 	plotly.offline.plot(fig,auto_open=True)

	# conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
	# plot_words(conv_tsne_embds, 0, 2000, 1)

	# glove_tsne_embds = TSNE(n_components=2).fit_transform(glove_emds)
	# plot_words(glove_tsne_embds, 0, 2000, 1)

	from gensim.models import Word2Vec
	import nltk
	nltk.download('punkt')
	data['tokenized'] = data.apply(lambda row : nltk.word_tokenize(row['text']), axis=1)
	print data.head()
	model_w2v = Word2Vec(data['tokenized'], size=100)
	X = model_w2v[model_w2v.wv.vocab]
	from sklearn.decomposition import TruncatedSVD
	from sklearn.decomposition import TruncatedSVD
	tsvd = TruncatedSVD(n_components=5, n_iter=10)
	result = tsvd.fit_transform(X)
	print result.shape

	tsvd_word_list = []
	words = list(model_w2v.wv.vocab)
	for i, word in enumerate(words):
		tsvd_word_list.append(word)
	trace = go.Scatter(
		x = result[0:len(tsvd_word_list), 0], 
		y = result[0:len(tsvd_word_list), 1],
		mode = 'markers',
		text= tsvd_word_list[0:len(tsvd_word_list)]
	)
	layout = dict(title= 'SVD 1 vs SVD 2',
		yaxis = dict(title='SVD 2'),
		xaxis = dict(title='SVD 1'),
		hovermode= 'closest')
	fig = dict(data = [trace], layout= layout)
	plotly.offline.plot(fig,auto_open=True)


main()