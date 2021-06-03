

import streamlit as st
import numpy as np
import pickle
import pandas as pd

model = pickle.load(open('car_price_prediction_model.pkl', 'rb'))

def predict_price(year, present_price, kms_driven, fuel_type, seller_type, Transmission, Owner):
    input=np.array([[year, present_price, kms_driven, fuel_type, seller_type, Transmission, Owner]])
    prediction=model.predict(input)
    pred='{0:.{1}f}'.format(prediction[0], 3)
    return float(pred)

def main():
    
    # st.title("Car Price Prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Car Price Prediction </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)


    _year = st.text_input("Year")
    _presentprice = st.text_input("Present_price")
    _kmsdriven = st.text_input("Kms_driven")
    _fueltype = st.selectbox("Fuel_type [cng: 0, diesel: 1, petrol: 2]", [0, 1, 2])
    _sellertype = st.selectbox("Teller_type [dealer: 0, individual: 1]", [0, 1])
    _transmission = st.selectbox("Transmission [automatic : 0 , Manual : 1]", [0, 1])
    _owner = st.selectbox("Owner", [0, 1, 3])

        

    if st.button("Predict"):
        output = predict_price(_year, _presentprice, _kmsdriven, _fueltype, _sellertype, _transmission, _owner)
        st.success(output)

if __name__=='__main__':
    main()


