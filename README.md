# OIBSIP_Task4_EmailSpamDetection

##  Objective
Build a machine-learning model that classifies email messages as **Spam** or **Ham (Not Spam)**.

##  Tools & Libraries
- Python, Jupyter Notebook
- pandas, numpy
- scikit-learn (TfidfVectorizer, MultinomialNB, Pipeline)
- matplotlib, seaborn
- joblib (to save the trained model)

## Steps Performed
1. Load & clean data (drop unused columns, handle encoding, map labels to 0/1).
2. Exploratory analysis (label distribution, message length).
3. Text preprocessing + TF‑IDF vectorization.
4. Train/test split with stratification.
5. Train **Multinomial Naive Bayes** via a scikit‑learn **Pipeline**.
6. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix.
7. Save trained pipeline as `spam_nb_pipeline.joblib`.
8. Provide a helper to test custom messages.

## ✅ Outcome
A robust spam detector that achieves high accuracy on the test set and can be used to score new messages.

