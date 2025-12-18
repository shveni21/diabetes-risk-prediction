# training_script.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib # Used for saving and loading models

def train_model():
    # Load the dataset
    df = pd.read_csv('fingerprint_diabetes_dataset.csv')

    # Drop the 'ID' column as it's not a feature
    df = df.drop('ID', axis=1)

    # Encode categorical features
    # Fingerprint_Type: arch, loop, whorl
    # Family_History: yes, no
    # Diabetes_Risk: low, medium, high (target variable)

    le_fingerprint = LabelEncoder()
    df['Fingerprint_Type'] = le_fingerprint.fit_transform(df['Fingerprint_Type'])
    joblib.dump(le_fingerprint, 'model/le_fingerprint.pkl') # Save encoder

    le_family_history = LabelEncoder()
    df['Family_History'] = le_family_history.fit_transform(df['Family_History'])
    joblib.dump(le_family_history, 'model/le_family_history.pkl') # Save encoder

    le_diabetes_risk = LabelEncoder()
    df['Diabetes_Risk'] = le_diabetes_risk.fit_transform(df['Diabetes_Risk'])
    joblib.dump(le_diabetes_risk, 'model/le_diabetes_risk.pkl') # Save encoder

    # Define features (X) and target (y)
    X = df[['Fingerprint_Type', 'Ridge_Count', 'Age', 'BMI', 'Family_History']]
    y = df['Diabetes_Risk']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model (optional, for checking performance)
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained model
    joblib.dump(model, 'model/diabetes_model.pkl')
    print("Model and encoders saved successfully in the 'model' directory.")

if __name__ == "__main__":
    train_model()