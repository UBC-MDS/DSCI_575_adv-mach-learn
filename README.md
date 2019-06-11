# DSCI 575: Advanced Machine Learning

Advanced machine learning methods, with an undercurrent of natural language processing (NLP) applications. Word embeddings, Markov chains, hidden Markov models, topic modeling, recurrent neural networks.


## Teaching team

| Position | Name  | Slack Handle | GHE Handle | Lab section | Office hour |
| :------: | :---: | :----------: | :--------: | :--------: | :--------: |
| Lecture instructor | Varada Kolhatkar | @Varada | @kvarada | all | Mondays 4:15 - 5:15pm at ESB 1045 |   
| Lab instructor  | Varada Kolhatkar | @Varada | @kvarada | all | Mondays 4:15 - 5:15pm at ESB 1045 | 
| Teaching assistant | Flora (Qiuyan) Liu | @Flora Liu | @floraliu | L01 | Fridays 11:00am – noon at ESB 1045| 
| Teaching assistant | Jie Xiang  | @Doris Xiang |  | L02 | Wednesdays noon – 1:00pm at ESB 1045| 
| Teaching assistant | Gursimran Singh | @msimar | @msimar | L03 | Fridays 3:00 – 4:00pm at ESB 1045| 


## Tentative schedule

| Lecture  | Topic  | Lecture notes | Resources and optional readings |
|-------|------------|-----------|-----------|
|   1   | Word vectors, word embeddings | [lecture1](lectures/lecture1.ipynb) | Word2Vec papers: <li>[Distributed representations of words and phrases and their compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)</li> <li>[Efficient estimation of word representations in vector space](https://arxiv.org/pdf/1301.3781.pdf)</li> <li>[word2vec Explained](https://arxiv.org/pdf/1402.3722.pdf)</li>|
|   2   | Using word embeddings, text preprocessing | [lecture2](lectures/lecture2.ipynb) | Pre-trained embeddings:  <li>[word2vec](https://code.google.com/archive/p/word2vec/)</li> <li>[GloVe](https://nlp.stanford.edu/projects/glove/) </li> <li>[fastText](https://fasttext.cc/docs/en/pretrained-vectors.html)</li>Bias in word embeddings:<li>[Debiasing Word Embeddings](http://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)</li>|
|   3   | Ngrams, POS tagging, Markov chains | [lecture3](lectures/lecture3.ipynb) | [Markov chains in action](http://setosa.io/ev/markov-chains/)|
|   4   | Language models, PageRank| [lecture4](lectures/lecture4.ipynb) | [Dan Jurafsky's videos on PageRank](https://www.youtube.com/playlist?list=PLaZQkZp6WhWzSy3WKExE7656jBxfXJh3I)|
|   5   | Hidden Markov models  | [lecture5](lectures/lecture5.ipynb) | [Rabiner HMM tutorial](https://www.cs.ubc.ca/~murphyk/Bayes/rabiner.pdf) |
|   6   | Topic modeling | [lecture6](lectures/lecture6.ipynb) | Dave Blei [video lecture](https://www.youtube.com/watch?v=DDq3OVp9dNA&t=98s), [paper](http://menome.com/wp/wp-content/uploads/2014/12/Blei2011.pdf) |
|   7   | Recurrent Neural Networks (RNNs) | [lecture7](lectures/lecture7.ipynb) | [The Unreasonable Effectiveness of Recurrent Neural  Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)|
|   8   | More on RNNs and wrap up |  | [Visual step-by-step explanation of LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)|

## Lab Assignments


| Lab    | Topics covered   | Due date |
|-----|-------------|----------|
| 1 |Word embeddings| 2019-03-23 18:00|
| 2 | Markov chains, language models | 2019-03-30 18:00|
| 3 | HMMs, topic modeling | 2019-04-06 18:00|
| 4 | RNNs | 2019-04-13 18:00|


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
* Jurafsky, D., & Martin, J. H. (2017). [Speech and language processing](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf).
* Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). [Deep learning (Vol. 1)](http://www.deeplearningbook.org/). Cambridge: MIT press. 
* [Jacob Eisenstein. Natural Language Processing](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)
* Goldberg, Y. (2017). Neural network methods for natural language processing. Synthesis Lectures on Human Language Technologies, 10(1), 1-309. 
* Bird, S., Klein, E., & Loper, E. (2009). Natural language processing with Python. O'Reilly Media, Inc. [[link](http://www.nltk.org/book/)].
