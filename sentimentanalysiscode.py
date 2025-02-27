import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
neg_url = "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.neg"
pos_url = "https://raw.githubusercontent.com/yoonkim/CNN_sentence/master/rt-polarity.pos"
negative_reviews = pd.read_csv(neg_url, header=None, names=["review"], encoding='latin-1', on_bad_lines="skip")
negative_reviews["label"] = 0
positive_reviews = pd.read_csv(pos_url, header=None, names=["review"], encoding='latin-1', on_bad_lines="skip")
positive_reviews["label"] = 1
data = pd.concat([positive_reviews, negative_reviews]).reset_index(drop=True)
data = data.dropna().reset_index(drop=True)
num_words = 10000  # Vocabulary size
max_len = 200  # Max sequence length
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(data["review"])
X = tokenizer.texts_to_sequences(data["review"])
X = pad_sequences(X, maxlen=max_len)
y = np.array(data["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
embedding_dim = 128
hidden_units = 64
model = Sequential([
    Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(hidden_units, return_sequences=False)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2, callbacks=[early_stopping]
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')
