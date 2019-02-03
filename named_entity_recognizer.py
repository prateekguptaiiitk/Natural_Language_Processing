import nltk
import nltk.data
from nltk.corpus import state_union

'''
    Named Entity Type and Examples:
    ORGANIZATION - Georgia-Pacific Corp., WHO
    PERSON - Eddy Bonte, President Obama
    LOCATION - Murray River, Mount Everest
    DATE - June, 2008-06-29
    TIME - two fifty a m, 1:30 p.m.
    MONEY - 175 million Canadian Dollars, GBP 10.40
    PERCENT - twenty pct, 18.75 %
    FACILITY - Washington Monument, Stonehenge
    GPE - South East Asia, Midlothian
'''

text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

tokenized = custom_sentence_tokenizer.tokenize(text)

#print(tokenized)
def process_content():
    try:
        for i in tokenized[5: ]:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(word)

            # recognizing named entity
            namedEnt = nltk.ne_chunk(tagged, binary=False)
            namedEnt.draw()

    except Exception as e:
        print(str(e))

process_content()


