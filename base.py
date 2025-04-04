# Imports (only necessary ones)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[["v1", "v2"]].rename(columns={"v1": "Target", "v2": "Text"})

# Encode target
le = LabelEncoder()
data["Target"] = le.fit_transform(data["Target"])  # ham=0, spam=1

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data["Text"], data["Target"], test_size=0.2, random_state=42)

# Pipeline with built-in cleaning from TfidfVectorizer
nb_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode',
        token_pattern=r'\b\w\w+\b'  # filters out punctuation and very short tokens
    )),
    ("nb", MultinomialNB())
])

# Train and evaluate
nb_pipeline.fit(X_train, y_train)
preds = nb_pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print("Classification Report:\n", classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Save the pipeline for Django
joblib.dump(nb_pipeline, "spam_classifier_pipeline.pkl")
