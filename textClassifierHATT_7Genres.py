import sklearn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from collections import defaultdict
import re
import tensorflow
from tensorflow import keras
from bs4 import BeautifulSoup
from tensorflow.keras import initializers, regularizers, constraints
import matplotlib

from matplotlib import interactive
from keras.callbacks import ReduceLROnPlateau
interactive(True)
matplotlib.use('Agg')
import nltk

from nltk.stem import WordNetLemmatizer

import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
STOPWORDS = set(stopwords.words('english'))
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras import initializers
from keras import optimizers

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, CuDNNLSTM, Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from keras.models import Model

from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras import initializers


max_features = 30000
maxlen_sentence = 60
maxlen_word = 10
batch_size = 32
embedding_dims = 100
epochs = 100
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
df = pd.read_csv('prepro.txt', sep='\t', header=None)
"""
def clean_text(text):
    
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df[1] = df[1].astype(str)
df[1] = df[1].apply(clean_text)
df[1] = df[1].str.replace('\d+', '')
"""
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
df[1] = df[1].astype(str)
df['tokens'] = df[1].map(tokenizer.tokenize)
lemmat = WordNetLemmatizer()
token_to_lem = {}
# initialise word count
token_count = 0

for lst in df['tokens']:
    # iterate through all tokens of song
    for token in lst:
        token_count += 1
        # check if token is in dictionary
        if token not in token_to_lem:
            # add token to dictionary
            token_to_lem[token] = lemmat.lemmatize(token)
df['lem'] = df['tokens'].map(lambda lst: [token_to_lem[token] for token in lst])


stop_words = set(stopwords.words('english'))
df['lem'] = [' '.join(map(str, l)) for l in df['lem']]
df['lem'] = df['lem'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

df['lem'] = df['lem'].str.findall('\w{4,}').str.join(' ')
df['lem'] = df['lem'].replace('\*','',regex=True)
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['lem'].values)

word_index = tokenizer.word_index
print(len(word_index))
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df['lem'].values)
print(X)
X = pad_sequences(X, maxlen=maxlen_sentence * maxlen_word)
Y = tf.keras.utils.to_categorical(df[0], num_classes=7)
print('Shape of data tensor:', X.shape)
print('Shape of data y:', Y.shape)

from nltk.tokenize import word_tokenize
x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state = 42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

x_train = x_train.reshape((len(x_train), maxlen_sentence, maxlen_word))
x_val = x_val.reshape((len(x_val), maxlen_sentence, maxlen_word))
x_test = x_test.reshape((len(x_test), maxlen_sentence, maxlen_word))
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape: ', x_test.shape)

# print('50 instances of x_val: ', x_val[:50])


GLOVE_DIR = "."
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index)+1, embedding_dims))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index)+1, embedding_dims))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[-1],),
                                 initializer=self.init,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name),
                                     shape=(input_shape[1],),
                                     initializer='zero',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ÃƒÅ½Ã‚Âµ to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim
        
class HAN(Model):
    def __init__(self,
                 maxlen_sentence,
                 maxlen_word,
                 max_features,
                 embedding_dims,
                 class_num=7,
                 last_activation='softmax'):
        super(HAN, self).__init__()
        self.maxlen_sentence = maxlen_sentence
        self.maxlen_word = maxlen_word
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        # Word part
        input_word = Input(shape=(self.maxlen_word,))
        x_word = Embedding(len(word_index) + 1, self.embedding_dims,weights=[embedding_matrix], input_length=self.maxlen_word)(input_word)
        x_word = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x_word)  # LSTM or GRU
        dropout_1 = Dropout(0.4)(x_word)
        x_word = Bidirectional(GRU(64, return_sequences=True))(dropout_1)  # LSTM or GRU
        x_word = Attention(self.maxlen_word)(dropout_1)
        model_word = Model(input_word, x_word)
        # Sentence part
        self.word_encoder_att = TimeDistributed(model_word)
        self.sentence_encoder = Bidirectional(CuDNNLSTM(32, return_sequences=True))  # LSTM or GRU
        self.dropout = Dropout(0.4)
        self.sentence_att = Attention(self.maxlen_sentence)
        # Output part
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 3:
            raise ValueError('The rank of inputs of HAN must be 3, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen_sentence:
            raise ValueError('The maxlen_sentence of inputs of HAN must be %d, but now is %d' % (self.maxlen_sentence, inputs.get_shape()[1]))
        if inputs.get_shape()[2] != self.maxlen_word:
            raise ValueError('The maxlen_word of inputs of HAN must be %d, but now is %d' % (self.maxlen_word, inputs.get_shape()[2]))
        x_sentence = self.word_encoder_att(inputs)
        x_sentence = self.sentence_encoder(x_sentence)
        dropout_2 = self.dropout(x_sentence)
        x_sentence = self.sentence_att(dropout_2)
        output = self.classifier(x_sentence)
        return output
        
print('Build model...')
opt = keras.optimizers.Adam(learning_rate=0.0001)

model = HAN(maxlen_sentence, maxlen_word, max_features, embedding_dims)
model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])
print('Train...')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.0000001)

early_stopping = EarlyStopping(monitor='val_accuracy', patience=13, mode='max')

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[reduce_lr],
          validation_data=(x_val, y_val))

print(model.summary())

score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_test_arg=np.argmax(y_test,axis=1)
Y_pred = np.argmax(model.predict(x_test),axis=1)
cfm = confusion_matrix(y_test_arg, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cfm)
plt.figure()
disp.plot()
plt.savefig('cfm.png')


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, label='Train accuracy')
plt.plot(epochs, val_acc, label='Val accuracy')
plt.title('Training & validation accuracy')
plt.legend()
plt.savefig('acc.png')

plt.figure()
plt.plot(epochs, loss, label='Train loss')
plt.plot(epochs, val_loss, label='Val loss')
plt.title('Training & validation loss')
plt.legend()
plt.show()

plt.savefig('loss.png')
