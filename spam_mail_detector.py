import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

data = pd.read_csv(
    "SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print(data.head())
print(data.shape)
print(data["label"].value_counts())

data["message"] = data["message"].str.lower()
print(data["message"].head())

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
print(stop_words)


def remove_stopwords(text):
    words = text.split()
    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)


data["message"] = data["message"].apply(remove_stopwords)
print(data["message"].head())

data["label"] = data["label"].map({"ham": 0, "spam": 1})
print(data["label"].head())

X = data["message"]
y = data["label"]

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)