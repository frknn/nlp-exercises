import  nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

#Alttaki satır, PunktSentenceTokenizer'ı train eder. Yine Bush'un konuşmasıyla train ettik.
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Örnek metini train edilmiş haline göre cümlelere ayırır.
tokenized = custom_sent_tokenizer.tokenize(sample_text)


#Her cümleyi, kelimelerine ayırıp öge isimlerine göre tagler.
def process_content():
    nnctr=0
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

        #   finding named entities
        #   namedEnt = nltk.ne_chunk(tagged)
        #   namedEnt.draw()

        #     for i in tagged:
        #         if i[1] == 'NN':  #GBush'un konuşmasındaki NN'leri bulur
        #             nnctr += 1
        #             print(i)
        # print(nnctr)

            # Chunking things
            # chunkGram = """Chunk: {<RB.>*<VB.?>*<NNP>+<NN>?}"""
            # Chink
            # chunkGram = """Chunk: {<.*>+}
            #                     }<VB.?|IN|DT|TO>+{"""
            # chunkParser = nltk.RegexpParser(chunkGram)
            # chunked = chunkParser.parse(tagged)
            #
            # chunked.draw()





    except Exception as e:
        print(str(e))

process_content()
