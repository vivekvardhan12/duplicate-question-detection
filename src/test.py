from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import re
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from keras.models import Model
from keras.layers import Input,Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import GlobalMaxPooling1D

stop_words = set(stopwords.words('english'))

def rem_html_tags(question):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', question)

def removeSpecialSymbols(question):
    question = re.sub('\W+',' ', question)
    question = question.strip()
    return question

train = pd.read_csv('dataset/dataset.csv',encoding='iso-8859-1',nrows=50000)
X = []
Y = []
label = 0
nondup = 0
for i in range(len(train)):
    title = train.get_value(i, 'Title')
    title = rem_html_tags(title)
    title = removeSpecialSymbols(title)
    title = title.lower()

    tags = train.get_value(i, 'Tags')
    tags = tags.strip().lower()
    tags = tags.lower()

    body = train.get_value(i, 'Body')
    body = body.strip().lower()
    body = body.lower()
    body = removeSpecialSymbols(body)
    body = body.replace('\n',' ')
    body = body.strip()
    if 'possible duplicate' in body:
        label = 1
    else:
        nondup = nondup + 1
        label = 0
    data = title
    arr = data.split(" ")
    msg = ''
    for k in range(len(arr)):
        word = arr[k].strip()
        if len(word) > 2 and word not in stop_words:
            msg+=word+" "
    texts = msg.strip()
    if label == 1:
        Y.append(label)
        X.append(texts)
    if nondup <= 300:
        Y.append((label))
        X.append(texts)

X = np.asarray(X)
Y = np.asarray(Y)
Y = to_categorical(Y)
print('Total questions found in dataset is : '+str(len(X))+"\n")
'''
cv = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True)
X = cv.fit_transform(X).toarray()
print(X)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print('Total train size : '+str(X.shape)+"\n")
'''

max_fatures = 500
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

model = Sequential()

embedding_layer = Embedding(500, 100, input_length=X.shape[1], trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X, Y, nb_epoch=100, batch_size=256)
y_predicted = model.predict(X_test, batch_size=256)
print(y_test)
print("=========================")
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print("=========================")
recall = recall_score(np.argmax(y_test, axis=1), y_predicted,average='macro') * 100
accuracy = accuracy_score(np.argmax(y_test, axis=1),y_predicted)*100
print(str(recall)+" "+str(accuracy))

train = pd.read_csv('dataset/test.csv',encoding='iso-8859-1',nrows=50000)
for i in range(len(train)):
    body = train.get_value(i, 'question')
    textdata = body
    mytext = [textdata]
    twts = tokenizer.texts_to_sequences(mytext)
    twts = pad_sequences(twts, maxlen=10, dtype='int32', value=0)
    sentiment = model.predict(twts,batch_size=256,verbose = 2)[0]
    print(sentiment)
    result = np.argmax(sentiment)
    print(result)
'''
model = Sequential()
model.add(Dense(512, input_shape=(X.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("model fitting - more complex convolutional neural network")
model.summary()
model.fit(X, Y, nb_epoch=100, batch_size=256)
y_predicted = model.predict(X_test, batch_size=256)
print(y_test)
print("=========================")
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print("=========================")
recall = recall_score(np.argmax(y_test, axis=1), y_predicted,average='macro') * 100
accuracy = accuracy_score(np.argmax(y_test, axis=1),y_predicted)*100
print(str(recall)+" "+str(accuracy))
'''
'''
embed_dim = 70
lstm_out = 70
max_fatures = 500
print(X)
print('Total questions found in dataset is : '+str(X.shape)+"\n")
model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())
batch_size = 256
model.fit(X, Y, epochs = 100, batch_size=batch_size, verbose = 1)
y_predicted = model.predict(X_test, batch_size=batch_size)
print(y_test)
print("=========================")
y_predicted = np.argmax(y_predicted, axis=1)
print(y_predicted)
print("=========================")
recall = recall_score(np.argmax(y_test, axis=1), y_predicted,average='macro') * 100
accuracy = accuracy_score(np.argmax(y_test, axis=1),y_predicted)*100
print(str(recall)+" "+str(accuracy))
'''

