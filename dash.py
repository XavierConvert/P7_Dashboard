import streamlit as st
import pandas as pd
import requests


#st.write("Hello World")

st.title('Credit prediction')

cid=st.number_input('Veuillez saisir la référence du crédit',min_value=100000, max_value=500000)

predict_btn = st.button('Prédire')

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='GET', headers=headers, url=model_uri, json=data_json) # si KO voir avec methode GET (cf API)
 
    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    FASTAPI_URI = 'http://127.0.0.1:8000/prediction/%s' % cid
    
    if predict_btn:
        data = cid
        #pred = None

        pred = request_prediction(FASTAPI_URI, data)#[0] #* 100000
        
        st.write(pred)


if __name__ == '__main__':
    main()    

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