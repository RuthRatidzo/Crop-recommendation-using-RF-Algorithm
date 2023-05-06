import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings

st.set_page_config(page_title="Crop Recommendation", layout='centered',
                   initial_sidebar_state="collapsed")


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation System 🌱🌾 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):
            st.write("""
            Crop recommendation is one of the most significant parts of precision agriculture. Numerous elements are taken into account while recommending crops. In order to address crop selection problems, precision agriculture aims to specify these criteria on a site-by-site basis. Even if "site-specific" methodology has increased performance, the results of the systems still need to be examined. Not all systems for precision agriculture are made the same way. However, since errors can lead to major material and financial loss in agriculture, it is crucial that suggestions offered are accurate and precise.
            """)
        '''
        ## How does it work ❓ 
        Fill in all the spaces with values from your nutrient result and the model will predict the most suitable crops to grow in a particular farm based on various parameters
        '''

    with col2:
        st.subheader(" Find the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosporus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        if st.button('Predict'):
            loaded_model = load_model('RandomForest.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write('''
		    ## Results 🔍 
		    ''')
            col1.success(f"{prediction.item().title()} are recommended for your farm.")
    # code for html ☘️ 🌾 🌳 👨‍🌾  🍃

    st.warning(
        "Note: This A.I application is for educational/demo purposes only and cannot be relied upon. ")
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
