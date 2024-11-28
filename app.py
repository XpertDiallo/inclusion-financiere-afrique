# app.py

import streamlit as st
import pandas as pd
import pickle

# Charger le modèle entraîné
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Prédiction de l'Inclusion Financière en Afrique")
st.subheader("Auteur: Hussein DIALLO")

# Champs de saisie pour les caractéristiques
country = st.selectbox("Pays", ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
location_type = st.selectbox("Type de localisation", ['Rural', 'Urban'])
cellphone_access = st.selectbox("Accès au téléphone portable", ['Yes', 'No'])
household_size = st.number_input("Taille du ménage", min_value=1, max_value=50, value=1)
age_of_respondent = st.number_input("Âge du répondant", min_value=16, max_value=100, value=16)
gender_of_respondent = st.selectbox("Genre du répondant", ['Male', 'Female'])
relationship_with_head = st.selectbox("Relation avec le chef de famille", [
    'Head of Household', 'Spouse', 'Child', 'Parent', 'Other relative', 'Other non-relatives', 'Dont know'])
marital_status = st.selectbox("Statut marital", [
    'Married/Living together', 'Divorced/Seperated', 'Widowed', 'Single/Never Married', 'Dont know'])
education_level = st.selectbox("Niveau d'éducation", [
    'No formal education', 'Primary education', 'Secondary education', 'Vocational/Specialised training', 'Tertiary education', 'Other/Dont know/RTA'])
job_type = st.selectbox("Type d'emploi", [
    'Farming and Fishing', 'Self employed', 'Formally employed Government', 'Formally employed Private', 'Informally employed', 'Remittance Dependent', 'Government Dependent', 'Other Income', 'No Income', 'Dont Know/Refuse to answer'])

# Préparation des données pour la prédiction
data = {
    'country': country,
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent': gender_of_respondent,
    'relationship_with_head': relationship_with_head,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type
}

# Conversion en DataFrame
input_data = pd.DataFrame([data])

# Bouton de prédiction
if st.button("Prédire"):
    # Effectuer la prédiction
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("L'individu est susceptible d'avoir ou d'utiliser un compte bancaire.")
    else:
        st.warning("L'individu n'est pas susceptible d'avoir ou d'utiliser un compte bancaire.")
