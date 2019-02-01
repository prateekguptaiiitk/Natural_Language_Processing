from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = "This is an example sentence for showing stop words filteration."

stop_words = set(stopwords.words("english"))

# all the stop words of englist language defined in nltk
# print(stop_words)

words = word_tokenize(text)

filtered_sentence = []

# long process:-
#for w in words:
#    if w not in stop_words:
#        filtered_sentence.append(w)
#

# short process:-
filtered_sentence = [w for w in words if not w in stop_words]

print(filtered_sentence)
