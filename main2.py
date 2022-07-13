import pandas as pd
import content
import readability
df = pd.read_pickle("RSDD.pickle")
print list(df)
print df.head(10)
df=content.addSyntacticFeatures(df)
d1=content.getLanguageFeatures(df)
df=content.addLexicalFeatures(df)
print "*************DF*************"
print df.head(10)

import afinnsenti
import labmt
df['text'] = df.apply(content.getTextFromRecord, axis=1)
df = afinnsenti.addEmotionalFeature(df)
df = labmt.addEmotionalFeature(df)
print df.head(10)

