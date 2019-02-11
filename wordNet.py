from nltk.corpus import wordnet

syns = wordnet.synsets("program")

# synset
print("Synset: ",syns[0].name())

# word
print("Word: ",syns[0].lemmas()[0].name())

# definition
print("Definition: ",syns[0].definition())

# examples
print("Examples: ",syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print("Synonyms are:-")
print(set(synonyms))
print("Antonyms are:-")
print(set(antonyms))

# comparing similarity of two words and their tenses by Wu and Palmer method

word1 = wordnet.synset('ship.n.01')
word2 = wordnet.synset('boat.n.01')
print("Similarity between words ship and boat = ", word1.wup_similarity(word2))

word3 = wordnet.synset('car.n.01')
print("Similarity between words ship and car = ", word1.wup_similarity(word3))

word4 = wordnet.synset('cat.n.01')
print("Similarity between words ship and cat = ", word1.wup_similarity(word4))

