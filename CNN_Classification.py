
import numpy as np
import data_helpers
from w2v import train_word2vec

from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate

# ---------------------- Parameters section -------------------

# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
num_of_doc_in_each_group=1000


# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 25

# Prepossessing parameters
sequence_length = 200#400
max_words = 1000#5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 2
context = 5

#
# ---------------------- Parameters end -----------------------


def load_data(num_of_doc_in_each_group):

    x, y, vocabulary, vocabulary_inv_list = data_helpers.load_data(num_of_doc_in_each_group)
    
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}

    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(num_of_doc_in_each_group*5))
    x = x[shuffle_indices]
    y = y[shuffle_indices]
    train_len = int(num_of_doc_in_each_group*5 * 0.8)
    x_train = x[:train_len]
    y_train = y[:train_len]
    x_test = x[train_len:]
    y_test = y[train_len:]

    return x_train, y_train, x_test, y_test, vocabulary_inv


# Data Preparation
print("Load data...")
x_train, y_train, x_test, y_test, vocabulary_inv = load_data(num_of_doc_in_each_group)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)


#-------------------------------------------------- Build model
input_shape = (sequence_length,)
model_input = Input(shape=input_shape)

# Static model does not have embedding layer
z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
z = Dropout(dropout_prob[0])(z)

#Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(5, activation="softmax")(z)

model = Model(model_input, model_output)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()

# Initialize weights with word2vec
weights = np.array([v for v in embedding_weights.values()])
print("Initializing embedding layer with word2vec weights, shape", weights.shape)
embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([weights])

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs,
          validation_data=(x_test, y_test), verbose=1)



score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

y_pre=model.predict(x_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])