
import numpy
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble


import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers


from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
import content

import math

import matplotlib.pyplot as plt

def getData(df1, df2):
    import readability
    df1 = readability.prepare(df1)
    df2 = readability.prepare(df2)
    if 'text' not in df1.columns:
        df1['text'] = df1.apply(content.getTextFromRecord, axis=1)
    if 'text' not in df2.columns:
        df2['text'] = df2.apply(content.getTextFromRecord, axis=1)
    df1['label'] = 'class1'
    df2['label'] = 'class2'
    df1 = df1[['text','label']]
    df2 = df2[['text','label']]
    trainDF = pd.concat([df1, df2])
    trainDF = trainDF.reset_index()
    del trainDF['index']
    trainDF = trainDF.reindex(numpy.random.permutation(trainDF.index))
    df_x=trainDF["text"]
    df_y=trainDF["label"]
    #split test and train data
    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df_x, df_y)
    #label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)
    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    count_vect.fit(trainDF['text'])
    # transform the training and validation data using count vectorizer object
    xtrain_count =  count_vect.transform(train_x)
    xvalid_count =  count_vect.transform(valid_x)
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
    tfidf_vect.fit(trainDF['text'])
    xtrain_tfidf =  tfidf_vect.transform(train_x)
    xvalid_tfidf =  tfidf_vect.transform(valid_x)
    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram.fit(trainDF['text'])
    xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
    xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
    tfidf_vect_ngram_chars.fit(trainDF['text'])
    xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
    xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

    # features=[]
    # features.append(('CV',xtrain_count,xvalid_count))
    # features.append(('Word level TF-IFD',xtrain_tfidf,xvalid_tfidf))
    # features.append(('N-Gram level TF-IFD',xtrain_tfidf_ngram,xvalid_tfidf_ngram))
    # features.append(('Char level TF-IFD',xtrain_tfidf_ngram_chars,xvalid_tfidf_ngram_chars))
    
    # models=[]
    # models.append(('NB',naive_bayes.MultinomialNB()))
    # models.append(('LR',linear_model.LogisticRegression()))


    def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    	# fit the training dataset on the classifier
    	classifier.fit(feature_vector_train, label)
    	# predict the labels on validation dataset
    	predictions = classifier.predict(feature_vector_valid)
    	if is_neural_net:
    		predictions = predictions.argmax(axis=-1)
    	tn,fp,fn,tp=metrics.confusion_matrix(valid_y, predictions).ravel()
    	return metrics.accuracy_score(predictions, valid_y), metrics.f1_score(predictions, valid_y), metrics.recall_score(predictions, valid_y), metrics.precision_score(predictions, valid_y), tp,tn,fp,fn

    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
    print "NB, Count Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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
    
    # Naive Bayes on Word Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_scor,tp,tn,fp,fn = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
    print "NB, WordLevel TF-IDF: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # Naive Bayes on Ngram Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "NB, N-Gram Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # Naive Bayes on Character Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print "NB, CharLevel Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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
    print "*********************"

    # Linear Classifier on Count Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
    print "LR, Count Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # Linear Classifier on Word Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
    print "LR, WordLevel TF-IDF: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # Linear Classifier on Ngram Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "LR, N-Gram Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # Linear Classifier on Character Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn  = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print "LR, CharLevel Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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
    print "************************"

 	# SVM Classifier on Count Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
    print "SVM, Count Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # SVM Classifier on Word Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
    print "SVM, WordLevel TF-IDF: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # SVM Classifier on Ngram Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
    print "SVM, N-Gram Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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

    # SVM Classifier on Character Level TF IDF Vectors
    accuracy,f1_score,recall_score,precision_score,tp,tn,fp,fn = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
    print "SVM, CharLevel Vectors: "
    print "Accuracy:",accuracy
    print "f1_score",f1_score
    print "recall_score",recall_score
    print "precision_score",precision_score
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
    print "********************"

