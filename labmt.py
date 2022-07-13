#We took labmt dataset http://neuro.compute.dtu.dk/wiki/LabMT
import pandas as pd
import math

labmt = pd.read_csv('labmt.txt', skiprows=2, sep='\t', index_col=0)
print labmt.happiness_average
average = labmt.happiness_average.mean()
happiness = (labmt.happiness_average - average).to_dict()
# print happiness
def score(text):
    if text is None:
        return None
    #if contains only empty spaces
    if len(text.strip())<=0:
        return None
    words = text.split()
    x=sum([happiness.get(word.lower(), 0.0) for word in words]) / math.sqrt(len(words))
    print "******************HAPPINESS AVERAGE******************"
    print words
    print x
    return x
def addEmotionalFeature(df):
    df['labmt'] = df['text'].apply(score)
    return df