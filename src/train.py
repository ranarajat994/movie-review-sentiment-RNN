# ============================================================
# Movie Review Sentiment Analysis - train.py
# Models: SimpleRNN, LSTM, GRU  (best one is saved)
# Run from project root:  python src/train.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
tf.random.set_seed(42)
np.random.seed(42)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "IMDB_Dataset.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. Hyperparameters (chosen from the EDA below)
# ------------------------------------------------------------
MAX_WORDS = 10000      # vocabulary size: top 10,000 words cover ~92% of all words
MAX_LEN = 200          # 85% of cleaned reviews are 200 words or shorter
OOV_TOKEN = "<OOV>"    # any word outside the top 10,000 becomes <OOV>
PADDING = "pre"        # zeros go at the START, so real words sit next to the final output
TRUNCATING = "pre"     # long reviews lose their START; the ending (the verdict) is kept
EMBEDDING_DIM = 64
RNN_UNITS = 64
DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 10


# ------------------------------------------------------------
# 4. Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())


# ------------------------------------------------------------
# 5. EDA
# ------------------------------------------------------------
print("\nNull values:")
print(df.isnull().sum())

print("\nSentiment count:")
print(df["sentiment"].value_counts())

print("\nDuplicate reviews:", df.duplicated().sum())

df["word_count"] = df["review"].str.split().str.len()
print("\nWords per review:")
print(df["word_count"].describe())

print("\nReviews with HTML <br /> tags:", round(df["review"].str.contains("<br").mean() * 100, 1), "%")

# Plot 1: sentiment balance
plt.figure(figsize=(6, 4))
df["sentiment"].value_counts().plot(kind="bar", color=["#2e7d32", "#c62828"])
plt.title("Sentiment distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "eda_sentiment_count.png"))
plt.close()

# Plot 2: review length
plt.figure(figsize=(8, 4))
plt.hist(df["word_count"], bins=60, range=(0, 1000), color="#1565c0")
plt.axvline(df["word_count"].median(), color="black", linestyle="--", label="median")
plt.title("Review length (words, before cleaning)")
plt.xlabel("Words")
plt.ylabel("Number of reviews")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "eda_review_length.png"))
plt.close()


# ------------------------------------------------------------
# 6. Remove duplicates
# ------------------------------------------------------------
df = df.drop_duplicates(subset=["review"])
print("\nShape after removing duplicates:", df.shape)


# ------------------------------------------------------------
# 7. Text cleaning
#    Stopwords = very common words with little meaning (the, is, and...).
#    BUT negation words ("not", "no", "didn't") are KEPT,
#    because "good" and "not good" mean opposite things.
# ------------------------------------------------------------
stop_words = set(stopwords.words("english"))

negation_words = {
    "not", "no", "nor", "but", "against", "very", "too", "only",
    "don", "isn", "wasn", "aren", "weren", "doesn", "didn",
    "hasn", "haven", "hadn", "won", "wouldn", "couldn", "shouldn",
}
stop_words = stop_words - negation_words

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    # lowercase
    text = text.lower()

    # remove HTML tags like <br />
    text = re.sub(r"<.*?>", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # expand n't -> not  (didn't -> did not)
    text = re.sub(r"n't", " not", text)

    # keep only letters (removes numbers, punctuation, emojis)
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords and 1-letter words, then lemmatize (movies -> movie)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words
             if word not in stop_words and len(word) > 1]

    return " ".join(words)


print("\nCleaning text... (takes 1-2 minutes)")
df["clean_review"] = df["review"].apply(clean_text)

print("\nBefore cleaning:")
print(df["review"].iloc[1][:300])
print("\nAfter cleaning:")
print(df["clean_review"].iloc[1][:300])

df["clean_word_count"] = df["clean_review"].str.split().str.len()
print("\nWords per review after cleaning:")
print(df["clean_word_count"].describe())
print("Reviews covered fully by MAX_LEN =", MAX_LEN, ":",
      round((df["clean_word_count"] <= MAX_LEN).mean() * 100, 1), "%")


# ------------------------------------------------------------
# 8. Encode the target: positive -> 1, negative -> 0
# ------------------------------------------------------------
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

X = df["clean_review"]
y = df["label"]


# ------------------------------------------------------------
# 9. Train test split
#    (done BEFORE the tokenizer, so test words don't leak into the vocabulary)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain reviews:", len(X_train))
print("Test reviews :", len(X_test))


# ------------------------------------------------------------
# 10. Tokenizer (fit on TRAIN only)
#     Feature selection: only the top MAX_WORDS words are kept as features.
# ------------------------------------------------------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train)

print("\nTotal unique words in train:", len(tokenizer.word_index))
print("Words kept as features     :", MAX_WORDS)


# ------------------------------------------------------------
# 11. Text -> sequences of numbers -> padded sequences
# ------------------------------------------------------------
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING)

y_train = np.array(y_train)
y_test = np.array(y_test)

print("\nX_train_pad shape:", X_train_pad.shape)
print("X_test_pad shape :", X_test_pad.shape)
print("\nExample padded sequence (last 20 numbers):")
print(X_train_pad[0][-20:])


# ------------------------------------------------------------
# 12. Early stopping: stop when validation loss stops improving
#     and go back to the best weights (prevents overfitting)
# ------------------------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)


# ------------------------------------------------------------
# 13. Model 1 - SimpleRNN
# ------------------------------------------------------------
print("\n========== SimpleRNN ==========")
rnn_model = Sequential()
rnn_model.add(Input(shape=(MAX_LEN,)))
rnn_model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM))
rnn_model.add(SimpleRNN(RNN_UNITS))
rnn_model.add(Dropout(DROPOUT))
rnn_model.add(Dense(32, activation="relu"))
rnn_model.add(Dense(1, activation="sigmoid"))

rnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
rnn_model.summary()

rnn_history = rnn_model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

rnn_test_loss, rnn_test_acc = rnn_model.evaluate(X_test_pad, y_test, verbose=0)
rnn_val_acc = max(rnn_history.history["val_accuracy"])
print("SimpleRNN best val accuracy:", round(rnn_val_acc, 4))
print("SimpleRNN test accuracy    :", round(rnn_test_acc, 4))


# ------------------------------------------------------------
# 14. Model 2 - LSTM
# ------------------------------------------------------------
print("\n========== LSTM ==========")
lstm_model = Sequential()
lstm_model.add(Input(shape=(MAX_LEN,)))
lstm_model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM))
lstm_model.add(LSTM(RNN_UNITS))
lstm_model.add(Dropout(DROPOUT))
lstm_model.add(Dense(32, activation="relu"))
lstm_model.add(Dense(1, activation="sigmoid"))

lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.summary()

lstm_history = lstm_model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

lstm_test_loss, lstm_test_acc = lstm_model.evaluate(X_test_pad, y_test, verbose=0)
lstm_val_acc = max(lstm_history.history["val_accuracy"])
print("LSTM best val accuracy:", round(lstm_val_acc, 4))
print("LSTM test accuracy    :", round(lstm_test_acc, 4))


# ------------------------------------------------------------
# 15. Model 3 - GRU
# ------------------------------------------------------------
print("\n========== GRU ==========")
gru_model = Sequential()
gru_model.add(Input(shape=(MAX_LEN,)))
gru_model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM))
gru_model.add(GRU(RNN_UNITS))
gru_model.add(Dropout(DROPOUT))
gru_model.add(Dense(32, activation="relu"))
gru_model.add(Dense(1, activation="sigmoid"))

gru_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
gru_model.summary()

gru_history = gru_model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

gru_test_loss, gru_test_acc = gru_model.evaluate(X_test_pad, y_test, verbose=0)
gru_val_acc = max(gru_history.history["val_accuracy"])
print("GRU best val accuracy:", round(gru_val_acc, 4))
print("GRU test accuracy    :", round(gru_test_acc, 4))


# ------------------------------------------------------------
# 16. Compare the three models
#     Best model is chosen by VALIDATION accuracy (not test),
#     so the test set stays an untouched, fair check.
# ------------------------------------------------------------
print("\n========== Comparison ==========")
print("SimpleRNN  -> val:", round(rnn_val_acc, 4), " test:", round(rnn_test_acc, 4))
print("LSTM       -> val:", round(lstm_val_acc, 4), " test:", round(lstm_test_acc, 4))
print("GRU        -> val:", round(gru_val_acc, 4), " test:", round(gru_test_acc, 4))

best_name = "GRU"
best_model = gru_model
best_history = gru_history
best_val = gru_val_acc

if lstm_val_acc > best_val:
    best_name = "LSTM"
    best_model = lstm_model
    best_history = lstm_history
    best_val = lstm_val_acc

if rnn_val_acc > best_val:
    best_name = "SimpleRNN"
    best_model = rnn_model
    best_history = rnn_history
    best_val = rnn_val_acc

print("\nBest model:", best_name)


# ------------------------------------------------------------
# 17. Final evaluation of the best model on train and test
# ------------------------------------------------------------
train_proba = best_model.predict(X_train_pad, verbose=0).ravel()
test_proba = best_model.predict(X_test_pad, verbose=0).ravel()

train_pred = (train_proba >= 0.5).astype(int)
test_pred = (test_proba >= 0.5).astype(int)

print("\n========== FINAL RESULT -", best_name, "==========")
print("Train accuracy:", round(accuracy_score(y_train, train_pred), 4))
print("Test accuracy :", round(accuracy_score(y_test, test_pred), 4))

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report (test):")
print(classification_report(y_test, test_pred, target_names=["negative", "positive"]))


# ------------------------------------------------------------
# 18. Plots: model comparison + training curves of the best model
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(["SimpleRNN", "LSTM", "GRU"], [rnn_test_acc, lstm_test_acc, gru_test_acc],
        color=["#9e9e9e", "#1565c0", "#2e7d32"])
plt.ylim(0.5, 1.0)
plt.title("Test accuracy by model")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"))
plt.close()

plt.figure(figsize=(11, 4))

plt.subplot(1, 2, 1)
plt.plot(best_history.history["accuracy"], label="train")
plt.plot(best_history.history["val_accuracy"], label="validation")
plt.title(best_name + " accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_history.history["loss"], label="train")
plt.plot(best_history.history["val_loss"], label="validation")
plt.title(best_name + " loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "training_curves.png"))
plt.close()


# ------------------------------------------------------------
# 19. Save the best model, tokenizer and settings for predict.py / app.py
# ------------------------------------------------------------
best_model.save(os.path.join(MODELS_DIR, "sentiment_model.keras"))

with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

settings = {
    "best_model": best_name,
    "max_words": MAX_WORDS,
    "max_len": MAX_LEN,
    "padding": PADDING,
    "truncating": TRUNCATING,
    "test_accuracy": round(accuracy_score(y_test, test_pred), 4),
}
with open(os.path.join(MODELS_DIR, "settings.pkl"), "wb") as f:
    pickle.dump(settings, f)

print("\nSaved model, tokenizer and settings in:", MODELS_DIR)
print("Saved plots in:", IMAGES_DIR)
