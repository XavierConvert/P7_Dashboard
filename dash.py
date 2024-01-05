import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import shap
shap.initjs()
from streamlit_shap import st_shap

st.title('Credit prediction')

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

  
def get_ids():
    IDS_URI='http://127.0.0.1:8000/ids'
    
    ids= request_ids(IDS_URI)
    cid = st.selectbox('Veuillez saisir la référence du crédit',ids)
    
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


def main():
    cid = get_ids()
    descr =get_global_data()
    #cl_data=get_client_data()
    PRED_URI = 'http://127.0.0.1:8000/prediction/%s' % cid
    SHAP_URI = 'http://127.0.0.1:8000/shap_val/%s' % cid
    CLIENT_URI='http://127.0.0.1:8000/client_details/%s' % cid
    
    
    
    predict_btn = st.button('Prédire')
    cb_shap=st.checkbox('Show SHAP values') 
           
    
    if predict_btn:
        data = cid
        #pred = None

        pred = request_prediction(PRED_URI, data)#[0] #* 100000
        
        st.write(pred['prediction'])
        st.write('Probabilité de remboursement (%):',round(pred['proba_rembour']*100,2))
        
        
        #shap_btn = st.button('Shap Values')
        
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
            
            st_shap(shap.plots.bar(exp))          
            #st.write(shap_val)
             
    cl_data=request_data(CLIENT_URI)
    cl_data=pd.DataFrame(cl_data)
    st.write('Données client')
    st.dataframe(cl_data,use_container_width=True)
    
    st.write('Stats globales - Tous dossiers:')
    descr=st.dataframe(descr,use_container_width=True)
    
if __name__ == '__main__':
    main()  
    
# streamlit run dash.py      

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