# Adapted from http://www.nltk.org/_modules/nltk/tag/hmm.html
from nltk.corpus import brown
import re 

from nltk.probability import (
    ConditionalFreqDist,
    ConditionalProbDist,
    DictionaryConditionalProbDist,
    DictionaryProbDist,
    FreqDist,
    LidstoneProbDist,
    MLEProbDist,
    MutableProbDist,
    RandomProbDist,
)
from nltk.tag.hmm import HiddenMarkovModelTrainer

def load_pos(num_sents):
    sentences = brown.tagged_sents(categories="news")[:num_sents]
    tag_re = re.compile(r"[*]|--|[^+*-]+")
    tag_set = set()
    symbols = set()

    cleaned_sentences = []
    for sentence in sentences:
        for i in range(len(sentence)):
            word, tag = sentence[i]
            word = word.lower()  # normalize
            symbols.add(word)  # log this word
            # Clean up the tag.
            tag = tag_re.match(tag).group()
            tag_set.add(tag)
            sentence[i] = (word, tag)  # store cleaned-up tagged token
        cleaned_sentences += [sentence]

    return cleaned_sentences, list(tag_set), list(symbols)

def demo_pos_supervised():
    # demonstrates POS tagging using supervised training

    print()
    print("HMM POS tagging demo")
    print()

    print("Training HMM...")
    labelled_sequences, tag_set, symbols = load_pos(20000)
    trainer = HiddenMarkovModelTrainer(tag_set, symbols)
    hmm = trainer.train_supervised(
        labelled_sequences[10:],
        estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins),
    )

    print("Testing...")
    hmm.test(labelled_sequences[:10], verbose=True)
    return hmm