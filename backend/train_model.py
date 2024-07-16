import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Sample data for illustration (replace with your actual dataset)
data = {
    "Gender": ["Male", "Female", "Male", "Female", "Female"],
    "Symptoms": ["Cough, Fever", "Headache, Nausea", "Back Pain", "Dizziness", "Cough, Sore Throat"],
    "Medical_History": ["Asthma", "Hypertension", "Diabetes", "None", "Asthma"],
    "Allergies": ["Penicillin", "None", "None", "Peanuts", "None"],
    "Medication": ["Med1", "Med2", "Med3", "Med4", "Med1"]
}

# Create DataFrame
df = pd.DataFrame(data)

# Preprocess data
label_encoders = {}
for column in ['Gender', 'Symptoms', 'Medical_History', 'Allergies', 'Medication']:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Split dataset
X = df.drop('Medication', axis=1)
y = df['Medication']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_accuracy = accuracy_score(y_test, nb_model.predict(X_test))

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))

# Select the best model (for simplicity, let's say Random Forest performed the best)
best_model = rf_model

# Save the best model
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Save label encoders
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

