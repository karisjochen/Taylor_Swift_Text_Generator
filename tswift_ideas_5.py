# Tokenizing words using tensorflow
from numpy.lib.function_base import extract
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
import spacy
from sklearn.utils import shuffle
import string
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from time import time

# LOAD MODEL AND TOKENIZER
# 20210326 is the 6 string model and 20210322 is the 25 string model
with open("saved_tsmodel/20210326tokenizer_tswift", "r") as infile: 
    token2_json = infile.read()
tokenizer = tokenizer_from_json(token2_json)
model = load_model('saved_tsmodel/20210326nn_tswift')
idx_word = tokenizer.index_word
model.summary()


# ********************************************************************************
# create model with 300 embeddings, 305 epochs, No validation set. Train on all songs
# ********************************************************************************

start_time = time()

df_ts = pd.read_excel('taylor_swift_full_20210325.xlsx', index_col=0)
# example on how to display full cell!! 
display(Markdown(df_ts.iloc[2]['lyrics']))
nlp = spacy.load('en_core_web_md')

seq_len = 6
subset = df_ts.iloc[:2]['lyrics']

# goal is to feed algorithm some tswift lyrics and ask it to return a desired
# number of words back. To do this, should I have a vectorized array where
# each row has a minimum of length 5 words and then adds one moer word until the entire song
# is in one row? I would obviousy have to pad them to fill to the max_length
# of her songs

# call model = number of times equal to number of desired output words

def clean_text(data):
    # returns list of songs in a cleaned up easier to parse version
    texts = []
    for song in data:
        song = song.lower()
        texts.append(song)

    return texts


def create_sequences(texts, train_len=6, lower=True):
    '''turn a set of texts into a sequnce of integers'''
    training_seq = []
    labels = []
    filters = '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n '
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts) # i think each sequence length is equal to song length

    # create a look-up ditionaries
    word_idx = tokenizer.word_index #returns word: index pair as a dict.
    idx_word = tokenizer.index_word #returns the index: word pair as a dict.
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts #shows the word counts for each dict in the entire corpous

    print(f'there are {num_words} unique words in the entire corpus')
    for seq in sequences:
        for i in range(train_len, len(seq)):
            extract = seq[i - train_len:i +1] # plus 1 so that we have the label which is the next word
            training_seq.append(extract[:-1])
            labels.append(extract[-1])
    print(f'There are {len(training_seq)} training sequences.')

    return tokenizer, word_idx, idx_word, num_words, word_counts, sequences, training_seq, labels


cleaned_songs = clean_text(df_ts['lyrics'])
#cleaned_songs = clean_text(subset)
tokenizer, word_idx, idx_word, num_words, word_counts, sequences, features, labels = create_sequences(cleaned_songs, train_len=seq_len, lower=True)
np.shape(features)
np.shape(labels)
# top 20 most common words
sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]


# randomly shufffle features and labels
features, labels = shuffle(features, labels)
X_train = np.array(features)
y_train = to_categorical(labels, num_classes=num_words, dtype=np.int8)

# CREATE AND TRAIN MODEL
# relu activation function says "return input or else return 0"
# I think this is important to have before the last layer bc of all the padded zeros
# softmax returns a probablity distribution for each of the classes

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    # output dim embeds each word a specified number of features, try setting this to 100 
    model.add(Embedding(input_dim=vocabulary_size, output_dim=300, input_length=seq_len))
    # a good idea to make number of neurons a multiple of sequence length; increase number units to increase accuracy
    # return  the sequences on all except last LSTM layer
    model.add(LSTM(units=64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)) 
    model.add(LSTM(64,  dropout=0.1, recurrent_dropout=0.1))
    # dense layer means to connect the layer coming in with the layer going out
    # adding more perceptrons than number of features will likely increase run time without increasing
    # much accuracy
    model.add(Dense(128, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

# define model
model = create_model(num_words, seq_len)
# FIT MODEL
# batch size is how many sequences to pass in at once. dr. jones recommends
# starting batch_size at 1% of data
# figure out howto make the callbacks work
#history = model.fit(X_train, y_train, batch_size=128, epochs=120,callbacks=callbacks,verbose=1, validation_data=(X_test, y_test))
history = model.fit(X_train, y_train, batch_size=128, epochs=350,verbose=1)


print('saving model')
# SAVE MODEL AND TOKENIZER
model.save('saved_tsmodel/20210326nn_tswift')
token_json = tokenizer.to_json()
# Writing to json 
with open("saved_tsmodel/20210326tokenizer_tswift", "w") as outfile: 
    outfile.write(token_json)
print('saving model done')



print('starting prediction!')

def beam_search(prediction,k):
    sequences = [[list(), 0.0]]
    all_candidates = list()
    # expand each current candidate
    for i in range(len(sequences)):
        seq, score = sequences[i]
        for j in range(len(prediction)):
            if prediction[j] > 0.0:
                candidate = [seq + [j], score - np.log(prediction[j])]
                all_candidates.append(candidate)
            else:
                continue

        # order all candidates by score
        ordered = sorted(all_candidates, key= lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]
    return sequences
            
def generate_text(model,tokenizer,seq_len,seed_text,num_gen_words, k):
    # remove punctuation
    nopunc1 = [char for char in seed_text if char not in string.punctuation]
    seed_text = ''.join(nopunc1)
    seed_text_list = seed_text.lower().split()
    seed_text = seed_text_list[:seq_len]

    all_candidates = list()
    all_candidates.append(seed_text)
    for i in range(num_gen_words):
        temp_candidates = []
        txt_list = [] 
        temp_sequences = []

        for row in all_candidates:
            
            
        # Take the input text string and encode it to a sequence
            encoded_text = tokenizer.texts_to_sequences([row])[0]
            pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
            prediction = model.predict(pad_encoded)[0]
            sequences = beam_search(prediction, k)
            #temp_sequences.append(sequences)
            temp_sequences = temp_sequences + sequences
        
            for z in sequences:
                word = z[0][0]
                word = idx_word[word]
                score = z[1]
                candidate = row + [word]
                txt_list.append([candidate, score])
    

        # order all candidates by score
        ordered = sorted(txt_list, key= lambda tup: tup[1])
        # select k best
        top_ordered = ordered[:k]


        for ordered in top_ordered:
            txt = ordered[0]
            #candidate = row + [txt]
            temp_candidates.append(txt)

        all_candidates = list()
        all_candidates = all_candidates + temp_candidates
        
    output = []
    for option in all_candidates:
        lyric =  ' '.join(option)
        output.append(lyric)
    return seed_text, output, all_candidates
    
idx_word = tokenizer.index_word

seed_text = "We could leave the Christmas lights up 'til January \
    And this is our place we make the rules And there's a dazzling haze\
        a mysterious way about you dear Have I known you 20 seconds or 20 years"

seed_text2 = 'Bottle extreme emotions Both sadness and happiness Overtime \
    Can marinate Like patient pickles Into rocket fuel'
seed_text3 =  'Like patient pickles Into rocket fuel'
input, output, predictions = generate_text(model,tokenizer,seq_len=6,seed_text=seed_text3,num_gen_words=15, k=5)



print('prediction done!')





finish_time = time()
print(f'Runtime was : {finish_time-start_time} seconds')
