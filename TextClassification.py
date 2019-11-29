import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf



# one-liner for getting documents
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]


# random.shuffle(documents)

# print(documents[1])

# all_words = []
# for w in movie_reviews.words():
#     all_words.append(w.lower())

short_pos=open("short_reviews/positive.txt","r").read()
short_neg=open("short_reviews/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append( (r,"pos") )

for r in short_neg.split('\n'):
    documents.append( (r,"neg") )


all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())



all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))
# print(all_words["chronic"])

word_features = [w[0] for w in all_words.most_common(5000)]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return  features

# print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(rev),category) for (rev, category) in documents]

random.shuffle(featuresets)

# print(featuresets[0])

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

# Loading classifier with pickle
# classifier_f = open("naivebayes.pickle","rb")
# classifier = pickle.load(classifier_f)
# classifier_f.close()


print("Original Naive Bayes alg accuracy: ", (nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

# saving classifier with pickle to reduce time
# save_classifier = open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()

# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Classifier alg accuracy: ", (nltk.classify.accuracy(MNB_classifier,testing_set))*100)

# Gaussian Naive Bayes
# GNB_classifier = SklearnClassifier(GaussianNB())
# GNB_classifier.train(training_set)
# print("GNB Classifier alg accuracy: ", (nltk.classify.accuracy(GNB_classifier,testing_set))*100)

# Bernoulli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB Classifier alg accuracy: ", (nltk.classify.accuracy(BNB_classifier,testing_set))*100)

# LogisticRegressionCV, SGDClassifier
# SVC,LinearSVC,NuSVC

LogisticRegressionCV_classifier = SklearnClassifier(LogisticRegressionCV())
LogisticRegressionCV_classifier.train(training_set)
print("LogisticRegressionCV Classifier alg accuracy: ", (nltk.classify.accuracy(LogisticRegressionCV_classifier,testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier alg accuracy: ", (nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC Classifier alg accuracy: ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Classifier alg accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier alg accuracy: ", (nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BNB_classifier,
                                  LogisticRegressionCV_classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier)

print("voted_classifier acc.%: ", (nltk.classify.accuracy(voted_classifier,testing_set))*100)

# print("Classification: ",voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
# print("Classification: ",voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
# print("Classification: ",voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
# print("Classification: ",voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
# print("Classification: ",voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
# print("Classification: ",voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)