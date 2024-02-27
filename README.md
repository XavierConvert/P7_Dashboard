# P7_Dashboard
Within DS/P7 project, repo dedicated to Dashboard (Streamlit)

One single branch (main)Repo dedicated to P7 Dashboard part

# Description

In this repo you'll find:

## dash.py

Code used in production with:

- routes to API (prod):
    - request_ids
    - request_data
    - request_prediction

- functions using routes to get API feedback (ids, predictions)
    - get_ids (using request_ids)
    - get_global_data (using request_data)

- predictions and shap_values are called directly through request_predictions


- construction of GUI with streamlit built_in functions => main()

+ API answers transformations

## dash_test.py

same than dash.py except routes (local instead of render production URI)

# Installation

For local access: 
- venv, 
- import libraries (requirements.txt)
- run: streamlit run dash_test.py

Production:

While pushing a new version of dash.py on Github, update of app

 https://xavierconvert-p7-dashboard.streamlit.app/

