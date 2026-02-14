# Sports vs. Politics Text Classifier

## 📌 Project Overview
This project implements a machine learning classifier to distinguish between text documents related to **Sports** and **Politics**. It compares six different algorithms to determine which is most effective for high-dimensional text data.

## 📂 Dataset
We used a subset of the **20 Newsgroups** dataset provided by Scikit-Learn:
- **Sports:** `rec.sport.baseball`, `rec.sport.hockey`
- **Politics:** `talk.politics.guns`, `talk.politics.mideast`, `talk.politics.misc`

**Total Documents:** ~4,600 (Train + Test)

## ⚙️ Methodology
- **Preprocessing:** Text cleaning (headers/footers removed), tokenization.
- **Features:** TF-IDF (Term Frequency-Inverse Document Frequency) with Unigrams and Bigrams.
- **Models Compared:**
  1. Multinomial Naive Bayes (Baseline)
  2. Logistic Regression
  3. Linear Support Vector Machine (SVM)
  4. Random Forest (Ensemble)
  5. K-Nearest Neighbors (KNN - Cosine Distance)
  6. Gradient Boosting

## 📊 Results
The models were evaluated on a held-out test set. **Multinomial Naive Bayes** achieved the highest accuracy.

| Model | Accuracy |
|-------|----------|
| **Naive Bayes** | **94.69%** |
| Linear SVM | 93.55% |
| Logistic Regression | 93.39% |
| K-Nearest Neighbors | 92.15% |
| Random Forest | 91.82% |
| Gradient Boosting | 86.35% |

### Confusion Matrices
The script automatically generates confusion matrices for each model. You can view them in the repository:
- `Naive_Bayes_cm.png`
- `Linear_SVM_cm.png`
- `Logistic_Regression_cm.png`
- `Random_Forest_cm.png`
- `K-Nearest_Neighbors_cm.png`
- `Gradient_Boosting_cm.png`

## 🚀 How to Run
1. **Install Dependencies:**
   ```bash
   pip install numpy scikit-learn matplotlib seaborn
   ```
2. **Run the script:**
   ```bash
   python classifier.py
   ```
   
