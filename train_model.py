import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import SMOTE   

# Load the dataset
def load_dataset():
    # Load from CSV
    dataset = pd.read_csv("malware.csv")
    X = dataset.drop("label", axis=1).values  # Features
    y = dataset["label"].values  # Labels
    return X, y

# Train the model
def train_model():
    # Load the dataset
    X, y = load_dataset()
    
    # Preprocessing: Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_scaled, y = smote.fit_resample(X_scaled, y)  # Balance the dataset
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize the RandomForest model
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    
    # Hyperparameter tuning using RandomizedSearchCV
    param_dist = {
        'n_estimators': [300, 500, 700],  # number of trees
        'max_depth': [30, 50, None],  # for deeper trees
        'min_samples_split': [2, 5, 10],  #for splitting
        'min_samples_leaf': [1, 2],  # Small leaf sizes for detail
        'max_features': ['sqrt', 'log2'],  # Focused feature selection
        'bootstrap': [True, False]  # Test with and without replacement
    }
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=15,  # Increase iterations for better tuning
        cv=6,       # 5-fold cross-validation
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    
    # Best model from randomized search
    best_model = random_search.best_estimator_
    
    # Test the model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the trained model and scaler
    joblib.dump(best_model, "malware_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Trained model and scaler saved.")

# Run the training
if __name__ =="_main_":
    try:
        train_model()
    except Exception as e:
        print(f"An error occurred during training: {e}")