from nltk.corpus import wordnet

# synonyms = wordnet.synsets("program")

# #synset
# print(synonyms[0].name())
#
# # just the word
# print(synonyms[0].lemmas()[0].name())
#
# # definition
# print(synonyms[0].definition())
#
# # examples
# print(synonyms[0].examples())

# finds syns and ants of a word 'good'
# synonyms = []
# antonyms = []
#
# for syn in wordnet.synsets("good"):
#     for l in syn.lemmas():
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
#
# print(set(synonyms))
# print(set(antonyms))

# finding similarity with Wu Palmer algorithm

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))
