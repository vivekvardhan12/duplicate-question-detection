
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

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

main = tkinter.Tk()
main.title('Duplicate Question Detection With Deep Learning in Stack Overflow')
main.geometry("1300x1200")


global filename
global model
global rnn_recall,cnn_recall,lstm_recall
global X,Y
global tokenizer
global X_train, X_test, y_train, y_test

stop_words = set(stopwords.words('english'))

def rem_html_tags(question):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', question)

def removeSpecialSymbols(question):
    question = re.sub('\W+',' ', question)
    question = question.strip()
    return question

def upload():
    text.delete('1.0', END)
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def word2Vec():
    text.delete('1.0', END)
    global tokenizer
    global X,Y
    global X_train, X_test, y_train, y_test
    train = pd.read_csv(filename,encoding='iso-8859-1',nrows=50000)
    X = []
    Y = []
    label = 0
    nondup = 0
    for i in range(len(train)):
        title = train._get_value(i, 'Title')
        title = rem_html_tags(title)
        title = removeSpecialSymbols(title)
        title = title.lower()

        tags = train._get_value(i, 'Tags')
        tags = tags.strip().lower()
        tags = tags.lower()

        body = train._get_value(i, 'Body')
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
    text.insert(END,'Total questions found in dataset is : '+str(len(X))+"\n")
    max_fatures = 500
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    text.insert(END,"Total questions used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total questions used for testing  : "+str(X_test.shape[0])+"\n")
    text.insert(END,"Total Vector size                 : "+str(X.shape[0])+",1481\n")
 

def WVRNN():
    text.delete('1.0', END)
    global rnn_recall
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
    print(model.summary())
    model.fit(X, Y, epochs=100, batch_size=256)
    y_predicted = model.predict(X_test, batch_size=256)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_temp = np.argmax(y_test, axis=1)
    recall = recall_score(y_temp, y_predicted,average='macro') * 100
    accuracy = accuracy_score(y_temp,y_predicted)*100
    text.insert(END,"WV-RNN Recall   : "+str(recall)+"\n")
    text.insert(END,"WV-RNN Accuracy : "+str(accuracy)+"\n\n")
    rnn_recall = recall
    

def WVCNN():
    global model
    global cnn_recall
    model = Sequential()
    embedding_layer = Embedding(500, 100, input_length=X.shape[1], trainable=False)
    model.add(embedding_layer)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    model.fit(X, Y, epochs=100, batch_size=256)
    y_predicted = model.predict(X_test, batch_size=256)
    y_predicted = np.argmax(y_predicted, axis=1)
    y_temp = np.argmax(y_test, axis=1)
    recall = recall_score(y_temp, y_predicted,average='macro') * 100
    accuracy = accuracy_score(y_temp,y_predicted)*100
    text.insert(END,"WV-CNN Recall   : "+str(recall)+"\n")
    text.insert(END,"WV-CNN Accuracy : "+str(accuracy)+"\n\n")
    cnn_recall = recall


def WVLSTM():
    global lstm_recall
    embed_dim = 70
    lstm_out = 70
    max_fatures = 500
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
    y_predicted = np.argmax(y_predicted, axis=1)
    y_temp = np.argmax(y_test, axis=1)
    recall = recall_score(y_temp, y_predicted,average='macro') * 100
    accuracy = accuracy_score(y_temp,y_predicted)*100
    text.insert(END,"WV-LSTM Recall   : "+str(recall)+"\n")
    text.insert(END,"WV-LSTM Accuracy : "+str(accuracy)+"\n\n")
    lstm_recall = recall

def detectDuplicates():
    text.delete('1.0', END)
    fname = filedialog.askopenfilename(initialdir="dataset")
    testfile = pd.read_csv(fname,encoding='iso-8859-1')
    for i in range(len(testfile)):
        body = testfile._get_value(i, 'question')
        textdata = body
        mytext = [textdata]
        twts = tokenizer.texts_to_sequences(mytext)
        twts = pad_sequences(twts, maxlen=10, dtype='int32', value=0)
        result = model.predict(twts,batch_size=256,verbose = 2)[0]
        result = np.argmax(result)
        print(result)
        msg = 'Master Question'
        if result == 1:
            msg = 'Non Master Question (duplicate)'
        text.insert(END,body+"======"+msg+"\n")    
        
                

def recallGraph():
    height = [rnn_recall,cnn_recall,lstm_recall]
    bars = ('WV-RNN Recall','WV-CNN Recall','WV-LSTM Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Duplicate Question Detection With Deep Learning in Stack Overflow')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Stack Overflow Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

vecButton = Button(main, text="Convert Dataset to Word2Vec", command=word2Vec)
vecButton.place(x=50,y=150)
vecButton.config(font=font1) 

rnnButton = Button(main, text="Run WV-RNN Algorithm", command=WVRNN)
rnnButton.place(x=50,y=200)
rnnButton.config(font=font1) 

cnnButton = Button(main, text="Run WV-CNN Algorithm", command=WVCNN)
cnnButton.place(x=50,y=250)
cnnButton.config(font=font1) 

lstmButton = Button(main, text="Run WV-LSTM Algorithm", command=WVLSTM)
lstmButton.place(x=50,y=300)
lstmButton.config(font=font1)

recallButton = Button(main, text="Recall Graph", command=recallGraph)
recallButton.place(x=50,y=350)
recallButton.config(font=font1)

detectButton = Button(main, text="Detect Duplicate Questions Test File", command=detectDuplicates)
detectButton.place(x=50,y=400)
detectButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=50,y=450)
exitButton.config(font=font1)

#main.config(bg='OliveDrab2')
main.mainloop()

