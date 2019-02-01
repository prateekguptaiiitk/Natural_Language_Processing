from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

# Stemming words
word_list = ["python", "pythoner", "pythoning", "pythoned", "pythonly"]

for w in word_list:
    print(ps.stem(w))

# Stemming sentence
text = "It is very important to be pythonly while you are pythoning with python. All the pythoners have pythoned poorly at least once"

words = word_tokenize(text)

for w in words:
    print(ps.stem(w))
