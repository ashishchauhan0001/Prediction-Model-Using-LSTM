import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Define the CustomLSTMCell class
class CustomLSTMCell(tf.keras.layers.Layer):
    def _init_(self, units, go_backwards=False):
        super(CustomLSTMCell, self)._init_()
        self.units = units
        self.go_backwards = go_backwards
        self.state_size = (units, units)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim + self.units, 4 * self.units), initializer='random_normal', trainable=True)
        self.recurrent_kernel = self.add_weight(shape=(self.units, 4 * self.units), initializer='random_normal', trainable=True)

    def sigmoid(self, x):
        return tf.math.sigmoid(x)

    def tanh(self, x):
        return tf.math.tanh(x)

    def call(self, inputs, states):
        h_tm1, c_tm1 = states
        x = inputs

        if self.go_backwards:
            x = tf.reverse(x, axis=[1])

        z = tf.matmul(tf.concat([h_tm1, x], axis=-1), self.kernel)
        z0, z1, z2, z3 = tf.split(z, 4, axis=-1)

        i = self.sigmoid(z0)
        f = self.sigmoid(z1)
        c_tilde = self.tanh(z2)
        o = self.sigmoid(z3)

        c = f * c_tm1 + i * c_tilde
        h = o * self.tanh(c)

        return h, [h, c]

    def get_config(self):
        return {'units': self.units, 'go_backwards': self.go_backwards}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Define the CustomBidirectional class
class CustomBidirectional(tf.keras.layers.Layer):
    def _init_(self, forward_layer, backward_layer, merge_mode='sum', **kwargs):
        super(CustomBidirectional, self)._init_(**kwargs)
        self.forward_layer = forward_layer
        self.backward_layer = backward_layer
        self.merge_mode = merge_mode

    def call(self, inputs, **kwargs):
        forward_outputs = self.forward_layer(inputs)
        backward_outputs = self.backward_layer(inputs)

        if self.merge_mode == 'sum':
            return forward_outputs + backward_outputs
        elif self.merge_mode == 'concat':
            return tf.concat([forward_outputs, backward_outputs], axis=-1)
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

    def compute_output_shape(self, input_shape):
        if self.merge_mode == 'sum':
            return self.forward_layer.compute_output_shape(input_shape)
        elif self.merge_mode == 'concat':
            forward_shape = self.forward_layer.compute_output_shape(input_shape)
            backward_shape = self.backward_layer.compute_output_shape(input_shape)
            return tf.TensorShape([forward_shape[0], forward_shape[1], forward_shape[2] + backward_shape[2]])
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

# Function to create LSTM model with Custom Bidirectional LSTM
def create_model(vocab_size, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=max_len, mask_zero=True))

    custom_lstm_fw = CustomLSTMCell(units=100, go_backwards=False)
    custom_lstm_bw = CustomLSTMCell(units=100, go_backwards=True)

    custom_bidirectional = CustomBidirectional(tf.keras.layers.RNN(custom_lstm_fw, return_sequences=True),
                                               tf.keras.layers.RNN(custom_lstm_bw, return_sequences=True),
                                               merge_mode='sum')

    model.add(custom_bidirectional)

    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Custom tokenizer function
def custom_tokenizer(sentences):
    word_index = {}
    index_word = {}
    for sentence in sentences:
        for word in sentence:
            if word not in word_index:
                idx = len(word_index) + 1
                word_index[word] = idx
                index_word[idx] = word

    # Add punctuation marks to word_index if not present
    punctuation_marks = [',', '?', '.', '!']
    for mark in punctuation_marks:
        if mark not in word_index:
            idx = len(word_index) + 1
            word_index[mark] = idx
            index_word[idx] = mark

    return word_index, index_word

# Function to load and preprocess data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    sentences = [sentence.strip().split() for sentence in data]
    return sentences

# Function to create input and target sequences with punctuation
def create_sequences_with_punctuation(sentences, punctuation_marks):
    X, y = [], []
    for sentence in sentences:
        X.append(sentence[:-1])
        y.append(sentence[1:] + [np.random.choice(punctuation_marks)])
    return X, y

# Generator function
def data_generator(data, word_index, max_len, batch_size, punctuation_marks):
    while True:
        for i in range(0, len(data), batch_size):
            X_batch, y_batch = create_sequences_with_punctuation(data[i:i+batch_size], punctuation_marks)
            X_batch = pad_sequences([[word_index[word] for word in seq] for seq in X_batch], maxlen=max_len, padding='post')
            y_batch = pad_sequences([[word_index[word] for word in seq] for seq in y_batch], maxlen=max_len, padding='post')
            yield X_batch, to_categorical(y_batch, num_classes=len(word_index) + 1)

# Function to restore punctuation
def restore_punctuation(model, word_index, max_len, unpunctuated_text):
    input_sequence = np.array([[word_index[word] for word in unpunctuated_text.split()]])
    input_sequence = pad_sequences(input_sequence, maxlen=max_len, padding='post')
    predicted_sequence = model.predict(input_sequence)
    predicted_sequence = np.argmax(predicted_sequence, axis=-1)

    # Convert indices back to words
    predicted_words = [index_word[idx] for idx in predicted_sequence[0] if idx != 0]

    # Reconstruct the punctuated text
    punctuated_text = ' '.join(predicted_words)
    return punctuated_text

# Load and preprocess data
train_data = load_data('train.txt')
test_data = load_data('test.txt')

# Tokenize data
word_index, index_word = custom_tokenizer(train_data)

vocab_size = len(word_index) + 1
max_len = 50  # Set a reasonable max_len value based on your data and available memory

# Split data into training and validation sets
split_index = int(0.9 * len(train_data))
train_data, val_data = train_data[:split_index], train_data[split_index:]

# Create and train the model using generators
batch_size = 16  # Reduce the batch size
punctuation_marks = [',', '?', '.', '!']
train_generator = data_generator(train_data, word_index, max_len, batch_size, punctuation_marks)
val_generator = data_generator(val_data, word_index, max_len, batch_size, punctuation_marks)

model = create_model(vocab_size, max_len)

# Introduce early stopping to monitor validation loss and stop training if it does not improve
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using generators
model.fit(train_generator, epochs=4, steps_per_epoch=len(train_data)//batch_size,
          validation_data=val_generator, validation_steps=len(val_data)//batch_size, callbacks=[early_stopping])

# Save the trained model
model.save('your_model_path.h5')

# Example usage for restoring punctuation
unpunctuated_text = "How are you are you good"
restored_text = restore_punctuation(model, word_index, max_len, unpunctuated_text)
print("Original Text:", unpunctuated_text)
print("Restored Text:", restored_text)