# ============================================================
# Movie Review Sentiment Analysis - predict.py
# Test from project root:  python src/predict.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import os
import re
import pickle
import warnings

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings("ignore")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


# ------------------------------------------------------------
# 3. Load the saved model, tokenizer and settings from train.py
# ------------------------------------------------------------
model = load_model(os.path.join(MODELS_DIR, "sentiment_model.keras"))

with open(os.path.join(MODELS_DIR, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

with open(os.path.join(MODELS_DIR, "settings.pkl"), "rb") as f:
    settings = pickle.load(f)

MAX_LEN = settings["max_len"]
PADDING = settings["padding"]
TRUNCATING = settings["truncating"]


# ------------------------------------------------------------
# 4. Same text cleaning as train.py
#    (must be EXACTLY the same, or the model sees different words)
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

    # keep only letters
    text = re.sub(r"[^a-z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords and 1-letter words, then lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words
             if word not in stop_words and len(word) > 1]

    return " ".join(words)


# ------------------------------------------------------------
# 5. Prediction function
# ------------------------------------------------------------
def predict_sentiment(review):

    # Step 1: clean the review
    cleaned = clean_text(review)

    # Step 2: words -> numbers using the saved tokenizer
    sequence = tokenizer.texts_to_sequences([cleaned])

    # Step 3: pad / truncate to the same length as training
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding=PADDING, truncating=TRUNCATING)

    # Step 4: predict (output = probability that the review is positive)
    probability = float(model.predict(padded, verbose=0)[0][0])

    if probability >= 0.5:
        sentiment = "Positive"
        confidence = probability
    else:
        sentiment = "Negative"
        confidence = 1 - probability

    return sentiment, confidence, probability, cleaned


# ------------------------------------------------------------
# 6. Quick test with a few sample reviews
# ------------------------------------------------------------
if __name__ == "__main__":

    print("Model used:", settings["best_model"])
    print("Test accuracy:", settings["test_accuracy"])

    review_1 = "This movie was absolutely wonderful. The acting was brilliant and the story kept me hooked till the end."
    sentiment, confidence, probability, cleaned = predict_sentiment(review_1)
    print("\nReview    :", review_1)
    print("Cleaned   :", cleaned)
    print("Prediction:", sentiment, "(" + str(round(confidence * 100, 1)) + "% confident)")

    review_2 = "Worst film I have seen in years. Boring plot, terrible acting and a complete waste of time."
    sentiment, confidence, probability, cleaned = predict_sentiment(review_2)
    print("\nReview    :", review_2)
    print("Cleaned   :", cleaned)
    print("Prediction:", sentiment, "(" + str(round(confidence * 100, 1)) + "% confident)")

    review_3 = "I didn't like this movie. It was not good at all."
    sentiment, confidence, probability, cleaned = predict_sentiment(review_3)
    print("\nReview    :", review_3)
    print("Cleaned   :", cleaned)
    print("Prediction:", sentiment, "(" + str(round(confidence * 100, 1)) + "% confident)")
