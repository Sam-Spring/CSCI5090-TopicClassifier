import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Load Data
df = pd.read_csv("TweetDataSplit/train_data.csv")

print("First 5 rows:\n", df.head())
print("\nColumns:", df.columns)
print("\nLabel distribution:\n", df['labels'].value_counts())

# Label names:
label_map = {
    0: "politics",
    1: "health",
    2: "technology",
    3: "entertainment",
    4: "money_finance",
    5: "relationship_dating",
    6: "education_learning",
    7: "work_careers",
    8: "science",
    9: "society_culture",
    10: "gamin",
    11: "lifestyle_hobbies",
    12: "sports",
    13: "automotive",
    14: "other"
}

target_names = list(label_map.values())

# Prepare Data
X = df['text']
y = df['labels']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Model 1: Naive Bayes
print("\n===== NAIVE BAYES =====")

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_preds = nb_model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, nb_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, nb_preds))


# M2: Logistic regression
print("\n===== LOGISTIC REGRESSION =====")

lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train_vec, y_train)

lr_preds = lr_model.predict(X_test_vec)

# Show sample predictions
print("\n===== SAMPLE PREDICTIONS =====\n")

for i in range(10):
    print("Post:", X_test.iloc[i])
    print("Actual:", y_test.iloc[i])
    print("Predicted:", lr_preds[i])
    print("-" * 50)

print("Accuracy:", accuracy_score(y_test, lr_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, lr_preds))


# Confusion Matrix
cm = confusion_matrix(y_test, lr_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)

plt.title("Confusion Matrix (Logistic Regression)")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()