import pandas as pd
import numpy as np
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
	print combinations
	for combination in combinations:
		
		if os.path.isfile(fname):
			f = open(fname, "r")
			rs = pickle.load(f)
			f.close()
		else:
			rs = {}

		if combination[0] in rs.keys():
			if combination[1] in rs[combination[0]].keys():
				print combination, "already exists, skipping..."
				continue

		print "doing", combination[0], "-", combination[1]
		df1 = pd.read_pickle(combination[0] + ".pickle")
		#by default index will be your row number
		#changes random index numbers to start from 0
		df1 = df1.reset_index()
		print df1.columns
		#applies the getTextFromRecord for each row of df since axis=1
		df1['text'] = df1.apply(content.getTextFromRecord, axis=1)
		#changes the index ordering to some random list of row numbers
		df1 = df1.reindex(numpy.random.permutation(df1.index))
		df2 = pd.read_pickle(combination[1] + ".pickle")
		print type(df2)
		df2 = df2.reset_index()
		df2['text'] = df2.apply(content.getTextFromRecord, axis=1)
		df2 = df2.reindex(numpy.random.permutation(df2.index))
		# keep only posts, keep only the text column
		df1 = df1[df1['parent_id'].astype(str)!='nan']
		#if any of the text column is NaN then drop such rows
		df1 = df1.dropna(subset=['text'])		
		df2 = df2[df2['parent_id'].astype(str)!='nan']
		df2 = df2.dropna(subset=['text'])
		print "After preprocessing the length of the Dataframe1:"+str(len(df1))+" length of Dataframe2:"+str(len(df2))
		results = []
		
		print "choosing min from", len(df1), len(df2)
		m = min(len(df1), len(df2))
		#for i in range(0,10):
		df1_min = df1.reindex(numpy.random.permutation(df1.index)).head(m)	
		df2_min = df2.reindex(numpy.random.permutation(df2.index)).head(m)
		import Cnn
		data = Cnn.getData(df1_min, df2_min)
		data['text']=data['text'].map(lambda x: Cnn.clean_text(x))	
		#extract only the text part of the dataframe with classification as class1-for depression and class2-for suicide
		data = classification.getData(df1_min, df2_min)
main()