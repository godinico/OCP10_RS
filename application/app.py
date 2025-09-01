import os
import streamlit as st
import requests

# URL de l'Azure Function (sans la clé)
BASE_FUNCTION_URL = "https://recommendation-function-ewcfc3cxcdhxcde2.westeurope-01.azurewebsites.net/api/recommendations"

# La clé est récupérée depuis une variable d'environnement
FUNCTION_KEY = os.getenv("AZURE_FUNCTION_KEY")

# Construire l'URL avec la clé
FUNCTION_URL = f"{BASE_FUNCTION_URL}?code={FUNCTION_KEY}"

# Liste d'exemple d'IDs utilisateurs
user_ids = [1, 2, 123, 456, 789, 101112, 131415]

st.title("Système de recommandation")

selected_user = st.selectbox("Choisissez un ID d'utilisateur", user_ids)
num_reco = st.number_input("Nombre d'articles à recommander", min_value=1, max_value=20, value=5, step=1)

if st.button("Affichage des recommandations"):
    params = {
        "user_id": selected_user,
        "num_recommendations": num_reco,
        "exclude_seen": "true"
    }
    response = requests.get(FUNCTION_URL, params=params)

    if response.status_code == 200:
        data = response.json()
        user_known = data["L'utilisateur est-il déjà connu ?"]
        articles = data["Numéros des articles recommandés"]

        if user_known == "Non":
            st.write("**Nouvel utilisateur ➡️ Suggestion des articles les plus populaires**")
        st.write("**Numéros des articles recommandés pour l'utilisateur choisi :**")
        st.write(articles)
    else:
        st.error(f"Erreur : {response.status_code} - {response.text}")
