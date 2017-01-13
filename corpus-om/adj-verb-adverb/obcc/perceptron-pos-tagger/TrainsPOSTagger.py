import nltk.corpus
import pickle
from nltk.corpus import mac_morpho
import sys

# corpus= nltk.corpus.mac_morpho.words()
# tagged = nltk.corpus.mac_morpho.tagged_words()
# print('len : ', len(corpus))
# print('fd : ',nltk.FreqDist(corpus))
# print(tagged[:50])
# i=0
# while i < 50:
#     if (tagged[i][1] == ('ART')):
#         print('artigos: ', tagged[i][0])
#     i+=1

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t

tsents = mac_morpho.tagged_sents()[:10]

tsents = [[(w.lower(),simplify_tag(t)) for (w,t) in sent if w] for sent in tsents if sent]

train = tsents[:9]
test = tsents[9:]

print test

tagger1 = nltk.PerceptronTagger(load=False)
tagger1.train(train)

print(tagger1.evaluate(test))

#print(tagger1.tag(['eu', 'gosto', 'de', 'comer', 'frutas']))

with open('PerceptronTagger.pickle','wb') as handle:
    pickle.dump(tagger1,handle)

tagger = pickle.load(open("PerceptronTagger.pickle"))
print(tagger.tag(["Eu", "comi", "frango"]))
