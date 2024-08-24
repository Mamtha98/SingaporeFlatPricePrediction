
import streamlit as st
from streamlit_option_menu import option_menu
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score


#page congiguration
st.set_page_config(page_title= "Singapore flat price prediction",
                   page_icon= 'random',
                   layout= "wide",)



st.markdown("<h1 style='text-align: center; color: black;'>SINGAPORE FLAT PRICE PREDICTION</h1>",
                unsafe_allow_html=True)

selected = option_menu(None, ["PREDICT RE SALE PRICE"],
                           icons=['cash-coin'],orientation='horizontal',default_index=0,
styles={
        "container": {"background-color":'white',"height":"60px","border": "3px solid #000000","border-radius": "0px"},
        "icon": {"color": "black", "font-size": "16px"}, 
        "nav-link": {"color":"black","font-size": "15px", "text-align": "centre", "margin":"4px", "--hover-color": "white","border": "1px solid #000000", },
        "nav-link-selected": {"background-color": "#5F259F"},})

if selected == 'PREDICT RE SALE PRICE':
    flat_type = ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '1 ROOM','MULTI-GENERATION']
    storey = ['10 TO 12', '01 TO 03', '04 TO 06', '07 TO 09', '13 TO 15',
       '19 TO 21', '22 TO 24', '16 TO 18', '34 TO 36', '28 TO 30',
       '37 TO 39', '49 TO 51', '25 TO 27', '40 TO 42', '31 TO 33',
       '46 TO 48', '43 TO 45']
    flat_model = ['Improved', 'New Generation', 'DBSS', 'Standard', 'Apartment',
       'Simplified', 'Model A', 'Premium Apartment', 'Adjoined flat',
       'Model A-Maisonette', 'Maisonette', 'Type S1', 'Type S2',
       'Model A2', 'Terrace', 'Improved-Maisonette', 'Premium Maisonette',
       'Multi Generation', 'Premium Apartment Loft', '2-room', '3Gen']
    c1,c2=st.columns([2,2])
    with c1:
        Flat_address=st.text_input('Enter the Flat address')
        Flat_block = st.text_input('Enter Block')
        Remaining_Lease_Days = st.text_input('Enter no of lease days available')
        log_floorsqm = st.text_input('Enter floor sqm')
    with c2:
        flat_type_value = st.selectbox('Flat Type', flat_type)
        storey_value = st.selectbox('Storey', storey)
        flat_model_value = st.selectbox('Flat Model', flat_model)
    
    with c1:
        st.write('')
        st.write('')
        st.write('')
        if st.button('PREDICT SELLING PRICE'):
            with open('flat_type.pkl', 'rb') as file:
                fte = pickle.load(file)
            with open('block.pkl', 'rb') as file:
                fbe = pickle.load(file)
            with open('storey_range.pkl', 'rb') as file:
                fse = pickle.load(file)
            with open('flat_model.pkl', 'rb') as file:
                fme = pickle.load(file)
            with open('Flat_Address.pkl', 'rb') as file:
                fae = pickle.load(file)
            with open('scalerx1.pkl', 'rb') as file:
                scaled_datax = pickle.load(file)
            with open('scalery1.pkl', 'rb') as file:
                scaled_datay = pickle.load(file)
            with open('dtreg_model1.pkl','rb') as file:
                dtreg_loaded_model = pickle.load(file)

            Remaining_Lease_Days = pd.to_numeric(Remaining_Lease_Days, errors='coerce')
            
            log_floorsqm = pd.to_numeric(log_floorsqm, errors='coerce')
            log_floorsqm = np.log(log_floorsqm)

            
            Flat_address_e=np.array([Flat_address])
            e_Flat_address = fae.transform(Flat_address_e)
            e_Flat_address =e_Flat_address[0].astype(int)

            Flat_block_e=np.array([Flat_block])
            st.write(Flat_block_e)
            e_Flat_block = fbe.transform(Flat_block_e)
            e_Flat_block =e_Flat_block[0].astype(int)

            flat_type_value_e=np.array([flat_type_value])
            e_flat_type_value = fte.transform(flat_type_value_e)
            e_flat_type_value =e_flat_type_value[0].astype(int)

            storey_value_e=np.array([storey_value])
            e_storey_value = fse.transform(storey_value_e)
            e_storey_value =e_storey_value[0].astype(int)

            flat_model_value_e=np.array([flat_model_value])
            e_flat_model_value = fme.transform(flat_model_value_e)
            e_flat_model_value =e_flat_model_value[0].astype(int)




           
            

            data =[]
            data.append(e_Flat_address)
            data.append(e_flat_type_value)
            data.append(e_Flat_block)
            data.append(e_storey_value)
            data.append(e_flat_model_value)
            data.append(Remaining_Lease_Days)
            data.append(log_floorsqm)
            x = np.array(data).reshape(1, -1)
            st.write(x)
            pred_model = scaled_datax.transform(x)
            price_predict= dtreg_loaded_model.predict(pred_model)
            y_pred_inverse_scaled = scaled_datay.inverse_transform(price_predict.reshape(-1, 1)).flatten()
            y_pred_original = np.exp(y_pred_inverse_scaled)
            predicted_price = str(y_pred_original)[1:-1]
            st.write(f'Predicted re sale value : :green[₹] :green[{predicted_price}]')