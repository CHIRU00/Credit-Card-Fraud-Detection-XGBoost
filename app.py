import numpy as np
import pandas as pd
import streamlit as st
import pickle

# Load the pre-trained model
with open('CreditCardFraudDetection.pkl', 'rb') as f:
    xgclf = pickle.load(f)

# Load the data
data = pd.read_csv('creditcard_2023.csv')

# Define columns to use as features
feature_cols = [f'V{i}' for i in range(1, 29)]

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write("Enter the following information to check if the transaction is legitimate or fraudulent:")

# Input fields
transaction_id = st.text_input('Transaction ID')
submit = st.button("Submit")

if submit:
    try:
        # Get the row corresponding to the transaction ID
        transaction_data = data.loc[data['id'] == int(transaction_id), feature_cols]
        
        if not transaction_data.empty:
            # Get the amount for the transaction ID
            transaction_amount = data.loc[data['id'] == int(transaction_id), 'Amount'].values[0]
            
            # Append the transaction amount as a feature
            transaction_data['Amount'] = transaction_amount
            
            # Make prediction
            prediction = xgclf.predict(transaction_data.values.reshape(1, -1))
            
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        else:
            st.write("Transaction ID not found in the dataset.")
    
    except:
        st.write("Please enter a valid transaction ID.")
