from gensim import models

class LabeledLineSentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        for uid, line in enumerate(open(self.filename)):
            yield LabeledLineSentence(words=line.split(), labels=['SENT_%s' % uid])

#LabeledLineSentence('/home/ozgecan/beyazperde.com/usercomments/clean/user_clean_0.txt')
sentence = LabeledLineSentence('/home/ozgecan/beyazperde.com/usercomments/clean/user_clean_0.txt', labels=["SENT_0"])
sentence1 = LabeledLineSentence('/home/ozgecan/beyazperde.com/usercomments/clean/user_clean_0.txt', labels=["SENT_1"])

sentences = [sentence, sentence1]
model = models.Doc2Vec(alpha=.025, min_alpha=.025, min_count=1)
model.build_vocab(sentences)

for epoch in range(10):
    model.train(sentences)
    model.alpha -= 0.002  # decrease the learning rate`
    model.min_alpha = model.alpha  # fix the learning rate, no decay

model.save("/home/ozgecan/my_model.txt")
model_loaded = models.Doc2Vec.load('/home/ozgecan/my_model.txt')

print
model.docvecs.most_similar(["SENT_0"])
print
model_loaded.docvecs.most_similar(["SENT_1"])