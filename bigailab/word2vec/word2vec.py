import gensim, logging
import os
from gensim.models.keyedvectors import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#global değişkenler
readFilePath = '/home/ozgecan/beyazperde.com/usercomments/clean'
writeFilePath = '/home/ozgecan/Desktop/usercomments_model.txt'
saveModelStatus = True
loadModelFilePath = '/home/ozgecan/Desktop/usercomments_model.txt'

#dosyayı  okumaya  sağlayan kısım
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        import os
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

#Modeli eğitmek için kullanılan method
class Training(object):
    def __init__(self,sentences):
        self.sentences= sentences
    def training(self):
        model = gensim.models.Word2Vec(self.sentences, size=100, window=5, min_count=5, workers=4)
        if writeFilePath:
         if saveModelStatus:
          model.save(writeFilePath)
         else :
          print("you didn't save your  model")
        else:
          print("you don't have anyfile for write model")

#Oluşturulan model dosyasının yüklenmesini sağlayan method
class LoadModelFile(object):
    if loadModelFilePath :
        model = gensim.models.Word2Vec.load(loadModelFilePath)
        print('################################################################')
        print('Similarity between iyi and güzel:')
        print(model.similarity('iyi', 'güzel'))
    else:
        print("you can't read modelFile")

class Test():
    sentences = MySentences(readFilePath)
    Training(sentences).training()
    LoadModelFile()
Test()


