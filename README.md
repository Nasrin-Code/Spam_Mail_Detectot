# Spam Mail Detector

An AI/ML project that classifies messages as **Spam** or **Ham (legitimate)** using Natural Language Processing and Machine Learning.

## Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn
- TF-IDF
- Naive Bayes

## Dataset

SMS Spam Collection dataset containing 5,572 messages labeled as `spam` or `ham`.

## Project Workflow

1. Load the dataset
2. Explore the data
3. Convert text to lowercase
4. Remove stopwords
5. Convert labels into numerical values
6. Split the data into training and testing sets
7. Convert text into TF-IDF features
8. Train a Multinomial Naive Bayes classifier
9. Predict spam and ham messages
10. Evaluate the model

## Model Performance

- Accuracy: **97.85%**
- Precision: **100%**
- F1 Score: **91.24%**

## Confusion Matrix

```text
[[966   0]
 [ 24 125]]