# Attribution: adapted from Jason Brownlee's book with permission

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import numpy as np
import sys
import os
import argparse
from collections import defaultdict, Counter
import string
import gc

def create_model(seq_length, n_vocab, layer_size=256, dropout_amount=0.5):
    """
    
    """
    model = Sequential()
    model.add(LSTM(layer_size, input_shape=(seq_length, n_vocab), return_sequences=True))
    model.add(Dropout(dropout_amount))
    model.add(LSTM(layer_size, return_sequences=True))
    model.add(Dropout(dropout_amount))
    model.add(LSTM(layer_size))
    model.add(Dropout(dropout_amount))
    model.add(Dense(n_vocab, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


def preprocess_data(filename, seq_length=100, vocab_size=100):
    
    with open(filename) as f:
        raw_text = f.read()
    # raw_text = raw_text.lower()
    # chars = sorted(list(set(raw_text)))
    print("length of raw text with all characters:", len(raw_text))
    
    char_count = Counter(raw_text)
    comn = char_count.most_common(vocab_size)
    vocabulary = [letter for letter, count in comn]
    try:
        print(vocabulary)
    except: # some issues on Windows
        pass
    # for ascii_letter in string.ascii_letters + string.digits:
    #     assert ascii_letter in vocabulary
    
    text = raw_text.translate({ord(c): None for c in char_count if c not in vocabulary}) 
    # (above) characters not in vocabulary are removed from the training string

    char_to_int = {c : i for i, c in enumerate(vocabulary)}
    int_to_char = {i : c for i, c in enumerate(vocabulary)}
    
    # summarize the loaded data
    n_vocab = len(vocabulary)
    print ("Total characters in text:", len(text))
    print ("Vocabulary size:", n_vocab)
    # prepare the dataset of input to output pairs encoded as integers
    print("Sequence length (aka context size, aka n-gram size):", seq_length)

    n_examples = len(text) - seq_length

    print ("Number of training examples:", n_examples)
    # blah
    dataX = np.zeros((n_examples, seq_length, n_vocab),dtype=bool)
    dataY = np.zeros((n_examples, n_vocab))
    
    for i in range(n_examples):
        seq_in = text[i:i + seq_length]
        char_out = text[i + seq_length]
        for j, char in enumerate(seq_in):
            dataX[i, j, char_to_int[char]] = 1.0
        dataY[i, char_to_int[char_out]] = 1.0
    
    return dataX, dataY, char_to_int, int_to_char

def train(filename, seq_length, batch_size=128):    
    dataX, dataY, char_to_int, int_to_char = preprocess_data(filename, seq_length=seq_length)
    
    n_examples, seq_length, n_vocab = dataX.shape

    model = create_model(seq_length, n_vocab)
    
    model.summary()
    
    # define the checkpoint
    prefix = os.path.split(filename)[-1]
    filepath= "weights-%s-%d-{epoch:02d}-{loss:.4f}.hdf5" % (prefix, seq_length)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # val_loss

    # fit the model
    model.fit(dataX, dataY, epochs=20, batch_size=batch_size, callbacks=[checkpoint], validation_split=0.20)
    model.fit_generator(generator, n_train_examples//batch_size, epochs=20, callbacks=[checkpoint], 
    validation_data=valid_generator, validation_steps=n_valid_examples//batch_size)


def gen(model, seed, n_vocab, int_to_char, temperature=1, num_gen=1000):
    print ("Seed:")
    for p in seed:
        sys.stdout.write(int_to_char[np.argmax(p)])
    print("\n--------------------------------------------\n")
    pattern = seed

    # generate characters
    for i in range(num_gen):
        prediction = model.predict(pattern[None], verbose=0)

        probabilities = prediction.flatten()
        if temperature != 1:
            # maybe not exactly right but close enough, and probably exactly right
            probabilities = probabilities**(1.0/temperature) 
            probabilities /= np.sum(probabilities)

        index = np.random.choice(n_vocab, p=probabilities)
            
        result = int_to_char[index]
        sys.stdout.write(result)
        sys.stdout.flush()

        new_char_one_hot = np.zeros(n_vocab)
        new_char_one_hot[index] = 1.0
        pattern = np.append(pattern[1:], new_char_one_hot[None], axis=0)
        
    print("\n")
    gc.collect() # http://stackoverflow.com/questions/40560795/tensorflow-attributeerror-nonetype-object-has-no-attribute-tf-deletestatus
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'gen'])
    parser.add_argument('filename', type=str)
    parser.add_argument('--weightsfile', type=str)
    parser.add_argument('-t', '--temperature', type=float, default=0.5)
    parser.add_argument('-n', '--contextlen', type=int, default=100)
    parser.add_argument('--genlen', type=int, default=500) 
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--primetext', type=str, default=None)
    
    args = parser.parse_args()

    if args.mode == "train":
        train(args.filename, seq_length=args.contextlen, batch_size=args.batchsize)
    else:
        if args.weightsfile is None:
                raise Exception("You need to provide a weights file for generation")
        
        # pick random seed (one of two ways depending on primetext)
        dataX, dataY, char_to_int, int_to_char = preprocess_data(args.filename, seq_length=args.contextlen)
        n_examples, seq_length, n_vocab = dataX.shape
        if args.primetext is None:
            # pick a random seed sequence from the training data
            start = np.random.randint(0, n_examples)
            seed = dataX[start]
        else:
            primetext = args.primetext[-args.contextlen:] # if you provide a primetext that is too long, just use the last contextlen characters
            seed = np.zeros((args.contextlen, n_vocab))
            c = 0
            for i in range(args.contextlen):
                if i >= args.contextlen-len(primetext):
                    seed[i, char_to_int[primetext[c]]] = 1.0
                    c += 1
                else:
                    seed[i,:] = 1/n_vocab # "uniform over all characters" (https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/sample.lua#L131)
            assert c==len(primetext)
    
        model = create_model(seq_length, n_vocab)
        model.load_weights(args.weightsfile)
        
        gen(model, seed, n_vocab, int_to_char, 
            temperature=args.temperature, num_gen=args.genlen)

    gc.collect() # http://stackoverflow.com/questions/40560795/tensorflow-attributeerror-nonetype-object-has-no-attribute-tf-deletestatus
    
