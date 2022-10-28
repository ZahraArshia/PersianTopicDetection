import data_helpers
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation , Dense, Dropout





import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    

def to_cat(y):
    new_y=[]
    for i in range(len(y)):
        if y[i]==0:
            new_y.append([1,0,0,0,0])
        if y[i]==1:
            new_y.append([0,1,0,0,0])
        if y[i]==2:
            new_y.append([0,0,1,0,0])
        if y[i]==3:
            new_y.append([0,0,0,1,0])
        if y[i]==4:
            new_y.append([0,0,0,0,1])
            
    new_y=np.array(new_y)
    return new_y



def shufle_Data(x,y,shuffle_indices):
    
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(len(x) * 0.8)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]
    
    y_train=to_cat(y_train)
    y_test=to_cat(y_test)
    
    return x_train,y_train,x_test,y_test

    

def my_loader(num_of_doc_in_each_group , vocab_size , mode):
    x, y = data_helpers.load_data_and_labels(num_of_doc_in_each_group)
    y = y.argmax(axis=1)

    # define Tokenizer with Vocab Size
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(x)
     
    x = tokenizer.texts_to_matrix(x, mode=mode)#tfidf , freq

    return x,y


# =============================================================================
# initial Vriable
# =============================================================================

num_of_doc_in_each_group=1000
num_labels = 5
vocab_size = 15000
batch_size = 100
epoch=20


# =============================================================================
# load frequency and shuffle Data
# =============================================================================
x_Freq,y_Freq=my_loader(num_of_doc_in_each_group , vocab_size , 'freq')
shuffle_indices = np.random.permutation(np.arange(len(y_Freq)))
x_train,y_train,x_test,y_test=shufle_Data(x_Freq , y_Freq , shuffle_indices)


# =============================================================================
# create frequency model
# =============================================================================
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_split=0.1)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('\n \n ------------------------------------------------------- \n')
print('Frequency Feature Extraction accuracy:', score[1])
   
plot_history(history)




#------------------------------------------------------------------------------


# =============================================================================
# load tfidf and shuffle Data
# =============================================================================
x_tfidf,y_tfidf=my_loader(num_of_doc_in_each_group , vocab_size , 'tfidf')
x_train,y_train,x_test,y_test=shufle_Data(x_tfidf,y_tfidf,shuffle_indices)


# =============================================================================
# create tfidf model
# =============================================================================
model = Sequential()
model.add(Dense(512, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epoch,
                    verbose=1,
                    validation_split=0.1)


score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('\n \n ------------------------------------------------------- \n')
print('tfidf Feature Extraction accuracy:', score[1])
   
plot_history(history)