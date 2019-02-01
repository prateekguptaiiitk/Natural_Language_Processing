import nltk
import nltk.data
from nltk.corpus import state_union

text = state_union.raw("2006-GWBush.txt")

custom_sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

tokenized = custom_sentence_tokenizer.tokenize(text)

#print(tokenized)
def process_content():
    try:
        for i in tokenized:
            word = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(word)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            #chunked.draw()

            # for getting both chunks and non-chunks
            for subtree in chunked.subtrees():
                print(subtree)
            
            # for filtering only chunks
            for subtree in chunked.subtrees(filter = lambda t: t.label() == "Chunk"):
                print(subtree)

    except Exception as e:
        print(str(e))

process_content()
