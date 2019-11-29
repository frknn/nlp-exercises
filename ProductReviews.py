import nltk
from nltk.corpus import stopwords
import random
from nltk.corpus import product_reviews_1,product_reviews_2
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import opinion_lexicon


positive = opinion_lexicon.positive()
sent1 = "The screen is awesome!"

sent1w = word_tokenize(sent1)
print(sent1w,len(sent1w))
for w in sent1w:
    if w in positive:
        print(w + ": Positive!")


chunkGram = r"""Chunk: {[..]} """

reviews = product_reviews_2.raw("Nokia_6600.txt")
documents = []
sentences = []

for rev in reviews.split('[t]'):
    documents.append(rev)
    for sent in rev.split('\n'):
        sentences.append(sent)


all_words = []

#print(documents[1])

# for s in sentences:
#     print(s)

#print(sentences)

for s in sentences:
    words = word_tokenize(s)
    for w in words:
        all_words.append(w.lower())

print(all_words)

for w in all_words:
    if w == 'phone' or w == '[' or w == ']' or w == '#' or w == """Chunk: {.\d}""":
        all_words.remove(w)

print(all_words)

stop_words = set(stopwords.words('english'))

wostopwords = [w for w in all_words if w not in stop_words]

print(wostopwords)


mostcommon = nltk.FreqDist(wostopwords).most_common(30)

print(mostcommon)

