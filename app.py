import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

import lime
from lime import lime_tabular
from urllib.request import urlopen
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
def main():
    # Chargement des données :

    df = pd.read_csv('data_balanced_2.csv')
    y = df['TARGET']
    df.drop(columns='TARGET', inplace=True)

    # Chargement du modèle :

    model = pickle.load(open('credit_final.pkl', 'rb'))
    liste_id = df['SK_ID_CURR'].values

    st.title('Dashboard Scoring Credit')
    st.markdown("Prédictions de scoring client, notre seuil de choix est de 50 %")
    hobby = st.selectbox(" Veuillez choisir un identifiant à saisir: ", liste_id)

    id_input = st.number_input(label='Veuillez saisir l\'identifiant du client demandeur de crédit:', format="%i",
                               value=0)
    if id_input not in liste_id:

        st.write("L'identifiant client n'est pas bon")


    elif (int(id_input) in liste_id):
        API_url = "http://127.0.0.1:5000/credit/" + str(id_input)
        with st.spinner('Chargement du score du client...'):
            json_url = urlopen(API_url)
            API_data = json.loads(json_url.read())
            classe_predite = API_data['prediction']
            pred_prob = API_data['proba']
            st.subheader('Le statut de la demande de crédit')
            st.write(classe_predite)

            st.subheader('Le pourcentage de scoring du client')
            st.write(round(pred_prob * 100, 0))

            X = df[df.SK_ID_CURR == id_input]
            X.drop(columns=['SK_ID_CURR'], inplace=True)

            dataframe = df.drop(['SK_ID_CURR'], axis=1)

            lime_explainer = lime_tabular.LimeTabularExplainer(dataframe.values,
                                                               feature_names=dataframe.columns,
                                                               class_names=list(y.unique()),
                                                               mode="classification")
            idx = X.index[0]
            exp = lime_explainer.explain_instance(X.loc[idx], model.predict_proba, num_features=10)

            components.html(exp.as_html(), height=800)
            exp.as_list()
if __name__ == "__main__":
    main()



