import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# ---------------------------------------------------------
# 1. DATA COLLECTION
# Using the standard 20 newsgroups dataset but filtering 
# only for the categories we actually care about.
# ---------------------------------------------------------

# defining the specific categories to fetch
cats = [
    'rec.sport.baseball', 'rec.sport.hockey',  # sports
    'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc' # politics
]

print("fetching dataset... might take a sec if not cached")

# grab train and test sets separately to avoid data leakage
train_data = fetch_20newsgroups(subset='train', categories=cats, remove=('headers', 'footers', 'quotes'))
test_data = fetch_20newsgroups(subset='test', categories=cats, remove=('headers', 'footers', 'quotes'))

print(f"training docs: {len(train_data.data)}")
print(f"testing docs: {len(test_data.data)}")

# ---------------------------------------------------------
# 2. PRE-PROCESSING & LABELS
# The dataset has specific labels for baseball, hockey etc.
# But we just want binary: 0 for Sport, 1 for Politics.
# ---------------------------------------------------------

# quick helper to map the fine-grained labels to our binary targets
# looking at the 'cats' list: indices 0,1 are sports, 2,3,4 are politics
def make_binary_labels(target_array):
    # if the label index is < 2 (0 or 1), it's sport (class 0)
    # otherwise it's politics (class 1)
    return np.where(target_array < 2, 0, 1)

y_train = make_binary_labels(train_data.target)
y_test = make_binary_labels(test_data.target)

class_names = ['Sport', 'Politics']

# sanity check to make sure classes aren't totally unbalanced
print(f"train split: {np.bincount(y_train)}")

# ---------------------------------------------------------
# 3. FEATURE EXTRACTION (TF-IDF)
# Using TF-IDF instead of simple counts to downweight 
# common words like 'the' or 'is'.
# ---------------------------------------------------------

# using n-grams (1,2) to catch phrases like "home run" or "white house"
# limiting features to 10k to keep the model lightweight
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')

# ---------------------------------------------------------
# 4. MODEL TRAINING & COMPARISON
# We are comparing NB, LogReg, and SVM.
# ---------------------------------------------------------

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(solver='liblinear'), # liblinear is usually decent for text
    "Linear SVM": LinearSVC(dual=False), # dual=False helps when n_samples > n_features
  
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # KNN (using cosine distance is usually better for text than euclidean)
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric='cosine'),
    
    # Gradient Boosting (Powerful but slower to train)
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    # build a pipeline: raw text -> tfidf -> model
    # this handles all the vectorization automatically
    pipe = make_pipeline(vec, model)
    
    # fit on training data
    pipe.fit(train_data.data, y_train)
    
    # predict on test data
    preds = pipe.predict(test_data.data)
    
    # evaluate
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    
    print(f"Accuracy: {acc:.4f}")
    print("Detailed Report:")
    print(classification_report(y_test, preds, target_names=class_names))

    # visualize confusion matrix for the report
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    filename = f"{name.replace(' ', '_')}_cm.png"
    
    # Save the figure to the current folder
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    
    # Close the plot so it doesn't pop up or consume memory
    plt.close()
# ---------------------------------------------------------
# 5. TEST WITH CUSTOM INPUT
# ---------------------------------------------------------

best_model_name = max(results, key=results.get)
print(f"\nBest performing model was: {best_model_name}")

# Retraining best model pipeline for the final demo
final_pipe = make_pipeline(vec, models[best_model_name])
final_pipe.fit(train_data.data, y_train)

def classify_doc(text):
    # Predict the class (0 or 1)
    prediction = final_pipe.predict([text])[0]
    
    # Get the actual model object (it's always the last step in the pipeline)
    model_step = final_pipe[-1]
    
    prob_str = "N/A"
    # Check if this specific model supports probability (SVM usually doesn't by default)
    if hasattr(model_step, "predict_proba"):
        # Get the probability of the predicted class
        probs = final_pipe.predict_proba([text])[0]
        max_prob = np.max(probs)
        prob_str = f"{max_prob:.2f}"
    
    label = class_names[prediction]
    print(f"\nInput: '{text}'")
    print(f"Classified as: {label} (Confidence: {prob_str})")

# demo
classify_doc("The government passed a new bill regarding tax reform.")
classify_doc("The quarterback threw a touchdown in the last quarter.")