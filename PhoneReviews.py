import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
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

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


phone_pos = open("phone_reviews/positive.txt", "r").read()
phone_neg = open("phone_reviews/negative.txt", "r").read()

documents = []

for r in phone_pos.split('\n'):
    documents.append( (r,"pos") )

for r in phone_neg.split('\n'):
    documents.append( (r,"neg") )


all_words = []

phone_pos_words = word_tokenize(phone_pos)
phone_neg_words = word_tokenize(phone_neg)

for w in phone_pos_words:
    all_words.append(w.lower())

for w in phone_neg_words:
    all_words.append(w.lower())

save_documents = open("phone_pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)

word_features = [w[0] for w in all_words.most_common(40)]

save_word_features = open("phone_pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featuresets = open("phone_pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()


random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[33:]
training_set = featuresets[:33]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

###############
save_classifier = open("phone_pickled_algos/originalnaivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

save_classifier = open("phone_pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_classifier = open("phone_pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_classifier = open("phone_pickled_algos/LogisticRegression_classifier5k.pickle", "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_classifier = open("phone_pickled_algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

##NuSVC_classifier = SklearnClassifier(NuSVC())
##NuSVC_classifier.train(training_set)
##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

save_classifier = open("phone_pickled_algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()