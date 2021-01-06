# DSCI 575: Advanced Machine Learning

Advanced machine learning methods in the context of natural language processing (NLP) applications. Word embeddings, Markov chains, hidden Markov models, topic modeling, recurrent neural networks.

2019/20 Instructor: **Varada Kolhatkar**

## High-level Course Learning Outcomes

By the end of the course, students are expected to be able to
- Explain and use word embeddings for word meaning representation. 
- Train your own word embedding and use pre-trained word embeddings. 
- Specify a Markov chain and carry out generation and inference with them. 
- Explain the general idea of stationary distribution in Markov chains.
- Explain hidden Markov models and carry out decoding with them. 
- Explain Latent Dirichlet Allocation (LDA) approach to topic modeling and carry out topic modeling on text data. 
- Explain Recurrent Neural Networks (RNNs) and use them for classification, generation, and image captioning.  

All videos are available [here](https://drive.google.com/drive/folders/1nMzTI-dNgkuitmqlHcndZ88zHeQhKel3)

## Tentative schedule

| Lecture  | Topic  | Notes | Resources and optional readings |
|-------|------------|-----------|-----------|
|   1   | Word vectors, word embeddings | [Notes](lectures/lecture1_word-embeddings.ipynb)| Word2Vec papers: <li>[Distributed representations of words and phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)</li> <li>[Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf)</li> <li>[word2vec Explained](https://arxiv.org/pdf/1402.3722.pdf)</li><li>[Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)</li>|
|   2   | Using word embeddings, text preprocessing | [Notes](lectures/lecture2_using-word-embeddings.ipynb) | <li>[Dan Jurafsky's video on tokenization](https://www.youtube.com/watch?v=pEwBjcYdcKw)</li>|
|   3   | Markov Models | [Notes](lectures/lecture3_Markov-chains.ipynb) | <li> [Markov chains in action](http://setosa.io/ev/markov-chains/) </li> <li> [Dan Jurafsky's videos on PageRank](https://www.youtube.com/playlist?list=PLaZQkZp6WhWzSy3WKExE7656jBxfXJh3I) </li> | 
|   4   | Hidden Markov models | [Notes](lectures/lecture4_HMMs.ipynb) | <li>[Nando de Freitas' lecture on HMMs](https://www.youtube.com/watch?v=jY2E6ExLxaw)</li> <li>[A gentle intro to HMMs by Eric Fosler-Lussier](http://di.ubi.pt/~jpaulo/competence/tutorials/hmm-tutorial-1.pdf)</li>|
|   5   | Topic modeling | [Notes](lectures/lecture5_Viterbi-topic-modeling.ipynb) | Dave Blei [video lecture](https://www.youtube.com/watch?v=DDq3OVp9dNA&t=98s), [paper](http://menome.com/wp/wp-content/uploads/2014/12/Blei2011.pdf) |
|   6   | Introduction to Recurrent Neural Networks (RNNs) | [Notes](lectures/lecture6_intro-to-RNNs.ipynb) | <li>[The Unreasonable Effectiveness of Recurrent Neural  Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)</li><li>[Sequence Processing with Recurrent Networks](https://web.stanford.edu/~jurafsky/slp3/9.pdf)</li>|  
|   7   | Long short term memory networks (LSTMs) | [Notes](lectures/lecture7_LSTMs.ipynb) | [Visual step-by-step explanation of LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
|   8   | Image captioning using CNNs and RNNs and wrap up | [Notes](lectures/lecture8_LSTMs-applications.ipynb) | [Jeff Heaton's video](https://www.youtube.com/watch?v=NmoW_AYWkb4&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN)|

## Resources
* [Google NLP API](https://cloud.google.com/natural-language/)
* [Stanford CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/syllabus.html)
* [LDA2vec: Word Embeddings in Topic Models](https://www.datacamp.com/community/tutorials/lda2vec-topic-model)
* [7 Types of Artificial Neural Networks for Natural Language Processing](https://www.kdnuggets.com/2017/10/7-types-artificial-neural-networks-natural-language-processing.html)
* https://distill.pub/
* [Model-Based Machine Learning](http://mbmlbook.com/toc.html)
* [RNNs in TensorFlow, a practical guide and undocumented features](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)
* [A list of readings about RNNs](https://github.com/tensorflow/magenta/tree/master/magenta/reviews)
* For NLP in R, see [Julia Silge's blog](https://juliasilge.com/blog/) posts on sentiment analysis of Jane Austen novels: [part 1](https://juliasilge.com/blog/you-must-allow-me/), [part 2](https://juliasilge.com/blog/if-i-loved-nlp-less/), [part 3](https://juliasilge.com/blog/life-changing-magic/), [part 4](https://juliasilge.com/blog/term-frequency-tf-idf/).
* [RNN resources](https://github.com/ajhalthor/awesome-rnn)

## Books
* Jurafsky, D., & Martin, J. H. [Speech and language processing](https://web.stanford.edu/~jurafsky/slp3/).
* Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep learning (Vol. 1)](http://www.deeplearningbook.org/). Cambridge: MIT press. 
* [Jacob Eisenstein. Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
* Goldberg, Y. (2017). Neural network methods for natural language processing. Synthesis Lectures on Human Language Technologies, 10(1), 1-309. 
* Bird, S., Klein, E., & Loper, E. (2009). [Natural language processing with Python](http://www.nltk.org/book/). O'Reilly Media, Inc.
