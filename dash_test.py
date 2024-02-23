import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import shap
shap.initjs()
from streamlit_shap import st_shap
#import plotly.graph_objects as go
#import matplotlib.pyplot as plt
import streamviz


###############################################
# Def des différentes requetes auprès de l'API:
###############################################

def request_ids(model_uri):
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        method='GET', headers=headers, url=model_uri)
    ids=response.json()
    
    return ids.values()
    
def request_data(model_uri):
    headers = {"Content-Type": "application/json"}
    response = requests.request(
        method='GET', headers=headers, url=model_uri)
    data=response.json()
    
    return data

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='GET', headers=headers, url=model_uri, json=data_json) # si KO voir avec methode GET (cf API)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

#####################################################################
# Def des différentes fonctions appelant différentes routes de l'API:
#####################################################################
  
def get_ids():
    IDS_URI='http://127.0.0.1:8000/ids'
    
    ids= request_ids(IDS_URI)
    cid = st.selectbox('Veuillez selectionner ou saisir la référence du client',ids)
    
    return cid

def get_global_data():
    DATA_URI='http://127.0.0.1:8000/data'
    data_desc=request_data(DATA_URI)
    data_desc=pd.DataFrame(data_desc)
       
    return data_desc

# def get_client_data():
    #cid = get_ids()
    #CLIENT_URI='http://127.0.0.1:8000/client_details/%s' % cid
    #cl_data=request_data(CLIENT_URI)
    #cl_data=pd.DataFrame(cl_data)
    #return cl_data

###############################
# Ecran principal du dashboard:
###############################
def main():
    st.set_page_config (layout="wide")

    col1, col2 =st.columns(2)

    with col1:
        st.image('Pret_a_depenser_logo.png',width=128)    

    with col2:
        
        st.title('Simulateur crédit')

    with st.sidebar.expander('A propos de cette app'):
        st.write("Ce dashboard a pour but de visualiser pour une référence client donnée, la probabilité de remboursement du crédit ainsi que les caractéristiques du souscripteur.")
    
    with st.sidebar.expander('Disclaimer'):    
        st.write("Les predictions affichées se basent sur un modèle d'apprentissage automatique.La demande devra être validée par un analyste crédit")
    st.divider()  
    
    cid = get_ids()
    descr =get_global_data()
    
        
    #cl_data=get_client_data()
    PRED_URI = 'http://127.0.0.1:8000/prediction/%s' % cid
    SHAP_URI = 'http://127.0.0.1:8000/shap_val/%s' % cid
    CLIENT_URI='http://127.0.0.1:8000/client_details/%s' % cid
    
    
    
    predict_btn = st.button('Prédire')
    cb_shap=st.checkbox("Afficher l'importance locale des variables") 
    
    if predict_btn:
        data = cid
        

        pred = request_prediction(PRED_URI, data)
        
        st.header(pred['prediction'])
        #st.write('Probabilité de remboursement (%):',round(pred['proba_rembour']*100,2))
        
        streamviz.gauge(pred['proba_rembour'],
                        gTitle=('Probabilité de remboursement: '),
                        sFix="%",
                        gSize='MED',
                        grMid=0.5
                        )
        
  
        if cb_shap:
            #data = cid
            shap_val=request_prediction(SHAP_URI,data)
            
            sv=[]
            fn=[]
            for k,v in shap_val.items():
                sv.append(v)
                fn.append(k)
            
            sv=np.array(sv)
            fn=np.array(fn)
            
            exp=shap.Explanation(sv,feature_names=fn)
            
            with st.expander('Explications'):
                st.text('Le graphe ci dessous représente les variables ayant le plus contribué à la prédiction\n\n'
                        'Les valeurs des variables bleues en bleu améliorent le score\n\n'
                        'Les valeurs des variables rougess en bleu détériorent le score'
                        )
            
            st_shap(shap.plots.bar(exp))          
                                     
    cl_data=request_data(CLIENT_URI)
    #cl_data=pd.DataFrame(cl_data)
    with st.expander('Données client (déroulez le menu pour les explications des variables):'):
        st.text('SK_ID_CURR:-------------référence du client\n'
                'FLAG_OWN_CAR:-----------le client possède t il son propre véhicule (0 =non)\n'
                'FLAG_OWN_REALTY:--------le client possède t il son propre bien immobilier (0 =non)\n'
                'AMT_INCOME_TOTAL:-------revenus du client\n'
                'AMT_CREDIT:-------------montant du crédit\n'
                'AMT_ANNUITY:------------montant des annuités\n'
                'AMT_GOODS_PRICE:--------montant du bien financé\n'
                'CNT_FAM_MEMBERS:--------nombre de personnes dans le foyer du client\n'
                "EXT_SOURCE_1, 2 et 3:---scores clients obtenus d'établissements de crédit (EC) tiers \n"
                "Prev_contract_nb:-------nb de crédits precedemment ouverts dabs nos livres ou dans d'autres EC\n"
                "Prev_AMT_CREDIT:--------montant total des crédits precedemment ouverts dans nos livres ou dans d'autres EC\n"
                'Refused_rate:-----------taux de refus sur demandes de crédits precédentes\n'
                'default_payment:--------le client a t-il déjà eu des incidents de paiements (0= non)\n'
                'INCOME_CREDIT_PERC:-----revenus du client / montant du crédit\n'
                "ANNUITY_INCOME_PERC:----montant de l'annuité / revenus du client \n"
                "PAYMENT_RATE:-----------montant de l'annuité / montant du crédit\n"
                'client_age:-------------age du client\n'
                'client_prof_exp:--------ancienneté professionnelle du client (en années)\n'
                'INCOME_PER_PERSON:------revenus par personne du foyer du client\n'
                "Cash_loans:-------------1= prêt à la consommation , 0= crédit revolving\n"
                'GENDER_FEMALE:----------genre (0 = M, 1= F)\n'
                'active_client:----------le client exerce-t-il une profession (0=non)\n'
                'relationship:-----------situation maritale du client (0: seul, 1: en couple)\n'
                )
        
    #st.write('Données client')
    st.dataframe(cl_data,use_container_width=True)
    
    st.write('Stats globales - Tous dossiers:')
    descr=st.dataframe(descr,use_container_width=True)
    
if __name__ == '__main__':
    main()  
  
#.venv\Scripts\activate.bat
    
# streamlit run dash_test.py

# Add a selectbox to the sidebar:
#add_selectbox = st.sidebar.selectbox(
#    'How would you like to be contacted?',
#    ('Email', 'Home phone', 'Mobile phone')
#)

# Add a slider to the sidebar:
#add_slider = st.sidebar.slider(
#    'Select a range of values',
#    0.0, 100.0, (25.0, 75.0)
#)