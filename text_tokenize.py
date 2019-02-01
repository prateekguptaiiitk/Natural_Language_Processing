from nltk.tokenize import sent_tokenize, word_tokenize

text = "Hello Mr. Gupta, How are you!. You look smart."

sentences = sent_tokenize(text)
words = word_tokenize(text)

print("The sentences in the text are:-")

for i in sentences:
    print(i)

print("The words in the text are:-")
for i in words:
    print(i)

