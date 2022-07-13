import pandas as pd
import numpy as np
import os
import spacy

#We took reference from https://cs.nyu.edu/grishman/jet/guide/PennPOS.html
#PartsOfSpeech tuple
#NN-noun,singular or mass
#PRP-Personal Pronoun
#IN-Preposition
#DT-Determiner
#RB-Adverb
#JJ-Adjective
#VB-Verb 
#CC-Coordinating conjuction
#NNS-noun,plural
#VBP-verb,past tense
PoS = [u'NN', u'PRP', u'IN', u'DT', u'RB', u'JJ', u'VB', u'CC', u'NNS', u'VBP']

#FirstPersonPronouns tuple
FirstPersonPronouns = ['i','me','mine','my','myself','our','ours','ourselves','us','we']

data_dir="/usr/local/lib/python2.7/dist-packages/spacy/data" #MAD ##MAD

#load the english language model
from spacy.lang.en import English
nlp = spacy.load('en_core_web_sm')

#nlp=English()
#nlp = English(data_dir=data_dir)
#nlp = spacy.load('en') ##
#nlp = spacy.load('/path/to/en') # unicode path



#get the body column of the panda data frame
#if body is NaN or deleted return None
def getTextFromRecord(row):
	if not pd.isnull(row['body']):
		rs = row['body']
	else:
		return None
		# rs = row['selftext']
	if rs=='[deleted]':
		return None
	return rs

def addLexicalFeatures(df):
	from collections import Counter
	df['text'] = df.apply(getTextFromRecord, axis=1)

	def cosineSimilarity(sentence1, sentence2, NounsOrPronounesOnly=False):
		import math
		if NounsOrPronounesOnly is False:
			vector1 = Counter([token.text.lower() for token in sentence1])
			vector2 = Counter([token.text.lower() for token in sentence2])
		else:
			vector1 = Counter([token.text.lower() for token in sentence1 if token.tag_.startswith("N") or token.tag_.startswith("PR")])
			vector2 = Counter([token.text.lower() for token in sentence2 if token.tag_.startswith("N") or token.tag_.startswith("PR")])			
		intersection = set(vector1.keys()) & set(vector2.keys())
		numerator = sum([vector1[x] * vector2[x] for x in intersection])
		sum1 = sum([vector1[x]**2 for x in vector1.keys()])
		sum2 = sum([vector2[x]**2 for x in vector2.keys()])
		denominator = math.sqrt(sum1) * math.sqrt(sum2)
		if not denominator:
			return 0.0
		else:
			return float(numerator) / denominator

	def getDocumentSimilarity(text, NounsOrPronounesOnly=False):
		doc = nlp(unicode(text))
		sentences = list(doc.sents)
		if len(sentences)<2:
			return None
		i = 0
		rs = 0
		for i in range(0, len(sentences)-1):
			rs += cosineSimilarity(sentences[i], sentences[i+1], NounsOrPronounesOnly)
		rs = rs/float(len(sentences)-1)
		return rs

	def getPronounsCounter(text):
		doc = nlp(unicode(text))
		sentences = list(doc.sents)
		from collections import Counter
		pronouns = []
		for sentence in sentences:
			pronouns.extend([token.text for token in sentence if token.tag_.startswith('PRP')])

		pronouns = Counter(pronouns)
		pronounsNo = np.sum(pronouns.values())
		return pd.Series({'sentencesNo':len(sentences),'pronouns':pronouns, 'pronounsNo': pronounsNo})

	def getDefiniteArticlesCounter(text):
		doc = nlp(unicode(text))
		sentences = list(doc.sents)
		from collections import Counter
		definiteArticles = []
		for sentence in sentences:
			definiteArticles.extend([token.text for token in sentence if (token.tag_ =='DT' and token.text.lower()=='the')])

		definiteArticles = Counter(definiteArticles)
		definiteArticlesNo = np.sum(definiteArticles.values())
		return pd.Series({'definiteArticlesNo': definiteArticlesNo})


	def getFirstPersonPronounsCounter(counter):
		l = [k for k in counter.keys() if k.lower() in FirstPersonPronouns]
		rs = {}
		for k in l:
			rs[k] = counter[k]
		firstPersonPronouns = Counter(rs)
		firstPersonPronounsNo = np.sum(firstPersonPronouns.values())
		return pd.Series({'firstPersonPronouns':firstPersonPronouns, 'firstPersonPronounsNo':firstPersonPronounsNo})

	df['documentSimilarity'] = df['text'].apply(getDocumentSimilarity)
	df['documentSimilarityNounsOrPronouns'] = df['text'].apply(getDocumentSimilarity, args=(True,))
	df[['pronouns','pronounsNo','sentencesNo']] = df['text'].apply(getPronounsCounter)
	df[['definiteArticlesNo']] = df['text'].apply(getDefiniteArticlesCounter)
	df[['firstPersonPronouns', 'firstPersonPronounsNo']] = df['pronouns'].apply(getFirstPersonPronounsCounter)
	df['firstPersonPronounsRatio'] = df['firstPersonPronounsNo']/df['pronounsNo'].astype(float)
	return df

i=0
def getSyntacticFeatures(row):

	text = getTextFromRecord(row)
	def getHeightToken(token):
		height = 1
		while token!=token.head:
			height +=1
			token = token.head
		return height

	def getVerbPhrasesLength(sentence):
		rs = [0]
		inVerb = False
		for token in sentence:
			if token.pos_=='VERB':
				if not inVerb:
					i = 1
				inVerb = True
				i +=1
			if token.pos_!='VERB':
				if inVerb:
					rs.append(i-1)
				inVerb = False
		return rs

	if text is None:
		return pd.Series({'maxHeight': np.nan, 'noun_chunks': np.nan, 'maxVerbPhraseLength': np.nan, 'subordinateConjuctions': np.nan}, dtype=object)
	
	#Apply spacy's NLP for the body
	doc = nlp(unicode(text))
	global i
	i+=1
	print i
	print "***************DOC CONTENT***************"
	print doc

	#retieve all the nouns in the body of all rows
	nouns=[]
	print "**************NOUNS*******************"
	for n in doc.noun_chunks:
		nouns.append(n.text)
	print nouns
	noun_chunks = len(list(doc.noun_chunks))


	#Seperate the body into a list of sentences
	sentences = list(doc.sents)

	maxHeight = 1
	subordinateConjuctions = 0

	print "**********SENTENCE WITH POS************"
	#POS tagging
	pos={}
	for sentence in sentences:
		print sentence
		for token in sentence:
			pos['token']=token
			pos['token_tag']=token.tag_
			print pos
		
		#find the ratio of prepositions in a sentence	
		subordinateConjuctions += len([token for token in sentence if token.tag_=='IN'])
	subordinateConjuctions = subordinateConjuctions/float(len(sentences))

	for sentence in sentences:
		height = 1
		if sentence.end==0:
			continue
		sentenceHeight = max([getHeightToken(token) for token in sentence])
		if maxHeight<sentenceHeight:
			maxHeight = sentenceHeight

	maxVerbPhraseLength = max(max([getVerbPhrasesLength(sentence) for sentence in sentences]))
	print pd.Series({'maxHeight':maxHeight-1, 'noun_chunks':noun_chunks, 'maxVerbPhraseLength': maxVerbPhraseLength, 'subordinateConjuctions':subordinateConjuctions}, dtype=object)

	return pd.Series({'maxHeight':maxHeight-1, 'noun_chunks':noun_chunks, 'maxVerbPhraseLength': maxVerbPhraseLength, 'subordinateConjuctions':subordinateConjuctions}, dtype=object)

def getLanguageFeatures(df):
	from collections import Counter
	import textblob
	import readability
	comments = df[df['parent_id'].astype(str)!='nan']
	posts = df[df['parent_id'].astype(str)=='nan']
	rs = Counter()
	# sentenceCount = 0
	# wordsCount = 0
	# wordsCounter = Counter()
	for t in comments[['body']].itertuples():
		o = textblob.TextBlob(t[1])
		rs = rs + Counter([t[1] for t in o.pos_tags])
		print "POS COUNTER FOR COMMENTS"
		print rs
		# sentenceCount = sentenceCount + len(o.raw_sentences)
	for t in posts[['selftext']].itertuples():
		o = textblob.TextBlob(t[1])
		rs = rs + Counter([t[1] for t in o.pos_tags])
		print "POS COUNTER FOR SELFTEXT"
		print rs
		# sentenceCount = sentenceCount + len(o.raw_sentences)

	# rs[wordsCounter] = np.mean(wordsCounter.values())/float(np.sum(wordsCounter.values()))
	# rs['numSentences'] = sentenceCount/float(len(df))

	total = float(np.sum(rs.values()))
	rs1 = {k : (100*rs[k])/total for k in rs.keys()}
	rs = pd.DataFrame(rs1.items())
	rs.columns = ['PoS', 'frequency']
	rs = rs.sort('frequency', ascending=False)
	rs = rs[rs['PoS'].isin(PoS)]
	d = rs.set_index('PoS')['frequency'].to_dict()
	d1 = {"PoS-"+k: d[k] for k in d.keys()}
	print "POS COUNTER DICTIONARY"
	print d1
	df = readability.prepare(df)
	df = readability.updateVocabularyFeatures(df)
	df1['LL'] = df['LL'].mean()
	return d1

def addSyntacticFeatures(df):
	df[['maxHeight','noun_chunks','maxVerbPhraseLength','subordinateConjuctions']] = df.apply(getSyntacticFeatures, axis=1)
	return df

def getURLtoPostRatio(df):
	def containedInURL(row):
		return row['permalink'] not in row['url']

	posts = df[df['parent_id'].astype(str)=='nan']
	cnt = len(posts)
	urls = len(posts[posts.apply(containedInURL, axis=1)])
	return urls/float(cnt)