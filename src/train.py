import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = 'data/processed/landmarks.csv'
MODEL_OUT = 'data/models/asl_model.pkl'

def train():
    # Load data
    df = pd.read_csv(DATA_FILE, header=None)
    X = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy() # sign labels

    print(f'Loaded {len(X)} samples across {len(set(y))} signs')

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, preds):.2%}')
    print(classification_report(y_test, preds))

    # Save
    os.makedirs('data/models', exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
    print(f'Model saved to {MODEL_OUT}')


if __name__ == '__main__':
    train()