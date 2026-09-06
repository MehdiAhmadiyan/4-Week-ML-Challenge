"""
Voting Classifiers Script.
This script demonstrates the "Wisdom of the Crowd" by training three completely
different models and combining them into an ensemble using a VotingClassifier.
It compares individual model scores against both Hard and Soft voting strategies.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_voting_classifiers():
    data_path = "data/moons_data.pkl"
    model_dir = "models"

    if not os.path.exists(data_path):
        print("Error: Moons data not found. Please import it first.")
        return

    print("Loading the Moons dataset...")
    moons_data = joblib.load(data_path)
    X_train, y_train = moons_data["X_train"], moons_data["y_train"]
    X_test, y_test = moons_data["X_test"], moons_data["y_test"]

    # Define Individual Models
    # Note: SVC requires probability=True to participate in Soft Voting
    log_clf = LogisticRegression(random_state=42)
    rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_clf = SVC(probability=True, random_state=42)

    # Define the Voting Classifier (Hard Voting)
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard' # Absolute democracy (majority rules)
    )

    # Train and Compare (Hard Voting)
    print("\nModel Accuracies (Hard Voting)")
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{clf.__class__.__name__:<25}: {acc * 100:.1f}%")

    print("\nConclusion: The VotingClassifier (Hard) usually beats the best individual model!")

    # Switch to Soft Voting
    print("\nSwitching to Soft Voting")
    # Soft voting averages the probabilities of all models, giving weight to highly confident ones.
    voting_clf.voting = "soft"
    voting_clf.fit(X_train, y_train)

    y_pred_soft = voting_clf.predict(X_test)
    soft_acc = accuracy_score(y_test, y_pred_soft)
    print(f"VotingClassifier (Soft):  {soft_acc * 100:.1f}%")
    print("Notice how Soft Voting pushes the accuracy even higher!")

    # Save the winning ensemble
    print("\nSaving the Soft Voting ensemble model...")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(voting_clf, os.path.join(model_dir, "soft_voting_clf.pkl"))
    print("Model saved successfully!")

if __name__ == "__main__":
    train_voting_classifiers()
