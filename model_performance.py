# model_performance.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def generate_model_performance():
    print("\n🔄 Generating Model Performance Metrics...")

    # Load dataset
    data = pd.read_csv("datasets/resume_data.csv")

    X_text = data["Skills"].astype(str)
    y = data["Job_Role"]

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_text)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # === Accuracy Graph ===
    train_acc = []
    test_acc = []
    iterations = list(range(100, 1100, 100))

    for i in iterations:
        temp_clf = LogisticRegression(max_iter=i)
        temp_clf.fit(X_train, y_train)

        train_acc.append(accuracy_score(y_train, temp_clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, temp_clf.predict(X_test)))

    plt.figure()
    plt.plot(iterations, train_acc, label="Training Accuracy")
    plt.plot(iterations, test_acc, label="Testing Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training vs Testing Accuracy Curve")
    plt.legend()
    accuracy_path = "accuracy_graph.png"
    plt.savefig(accuracy_path)
    plt.close()

    # === Confusion Matrix ===
    y_pred = clf.predict(X_test)
    valid_labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=valid_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=valid_labels, yticklabels=valid_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print("\n✅ Performance Metrics Generated Successfully!")
    print(f"📈 Accuracy Graph saved as: {accuracy_path}")
    print(f"🧩 Confusion Matrix saved as: {cm_path}")

if __name__ == "__main__":
    generate_model_performance()
