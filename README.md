# Doc2Vec
 Two models here: cbow ( continuous bag of words) where we use a bag of words to predict a target word and skip-gram where we use one word to predict its neighbors. 
 
 After this idea is proved to be effective and helpful, say, you can easily cluster and find similar words in a huge corpus, people then began thinking further: is it possible to have a higher level of representation on sentences, paragraphs or even documents.
 
 
Models

Similarly, there are two models in doc2vec: dbow and dm.

dbow (distributed bag of words)
It is a simpler model that ignores word order and training stage is quicker. The model uses no-local context/neighboring words in predictions. You see it is not considering the order of the words. From the paper [4], the figure below shows dbow.

dm (distributed memory)
We treat the paragraph as an extra word. Then it is concatenated/averaged with local context word vectors when making predictions. During training, both paragraph and word embeddings are updated. It calls for more computation and complexity.

