import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Load data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    
    return clf, le, x.columns, training, testing

# Load additional data from CSV files
def load_csv_data(file_path):
    data_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > 1:  # Ensure there are enough columns in the row
                key = row[0]
                values = row[1:]
                data_dict[key] = values
            else:
                logging.warning(f"Skipping row due to insufficient data: {row}")
    return data_dict

# Predict the disease based on selected symptoms
def predict_symptoms(clf, le, symptoms, feature_names):
    input_vector = np.zeros(len(feature_names))
    for symptom in symptoms:
        if symptom in feature_names:
            input_vector[feature_names.get_loc(symptom)] = 1
    prediction = clf.predict([input_vector])
    disease = le.inverse_transform(prediction)[0]
    return disease

# Streamlit application
def main():
    # Load model and data
    clf, le, feature_names, training, testing = load_data()
    description_dict = load_csv_data('symptom_Description.csv')
    precaution_dict = load_csv_data('symptom_precaution.csv')

    # Title
    st.title("Healthcare Chatbot")

    # Symptoms input
    st.subheader("Input Symptoms")
    symptoms = st.multiselect(
        "Select the symptoms you are experiencing:",
        options=feature_names
    )
    
    # Predict disease when the button is clicked
    if st.button("Predict"):
        if symptoms:
            predicted_disease = predict_symptoms(clf, le, symptoms, feature_names)
            
            st.subheader(f"Predicted Disease: {predicted_disease}")
            
            # Display description
            description = description_dict.get(predicted_disease, ["Description not available"])[0]
            st.subheader("Description")
            st.write(description)
            
            # Display precautions
            precautions = precaution_dict.get(predicted_disease, ["Precautions not available"])
            st.subheader("Precautions")
            for i, precaution in enumerate(precautions, start=1):
                st.write(f"{i}. {precaution}")
        else:
            st.write("Please select at least one symptom.")

# Run the Streamlit application
if __name__ == '__main__':
    main()
