import streamlit as st
import pandas as pd
import joblib

model = joblib.load("house.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
num_col = joblib.load("num_col.pkl")

st.title("House Price Prediction")
st.write("enter house details to predict price")



Area_SqFt = st.number_input("Area_SqFt",
                           min_value = 0.0,
                           value = 2777.141066)

Rooms = st.number_input("Rooms",
                        min_value = 1,
                        max_value = 7,
                        value = 5)

Build_Year = st.number_input("Build_Year",
                             min_value = 1970,
                             max_value = 2024,
                             value = 2021)

Location = st.selectbox("Location",
                        ["Jaipur",
                         "Indore",
                         "Lucknow",
                         "Kanpur",
                         "Noida",
                         "Delhi",
                         "Prayagraj",
                         "Gurugram",
                         "Jaipur"])

Street_Type = st.selectbox("Street_Type",
                           ["Residential Lane",
                            "Corner Plot",
                            "Highway Facing",
                            "Main Road",
                            "Gated Society"])

Furnishing = st.selectbox("Furnishing",
                          ["Semi-Furnishing",
                            "Furnishing",
                            "UnFurnishing"])

Property_Type = st.selectbox("Property_Type",
                             ["Apartment",
                              "Duplex",
                              "Villa",
                              "Independent"])

Has_Pool = st.selectbox("Has_Pool",
                        ["Yes",
                         "No"])

if st.button("Price"):

    encoded_columns = ["Area_SqFt","Rooms","Build_Year", "Location", "Street_Type", "Furnishing", "Property_Type", "Has_Pool"]

    input_data = pd.DataFrame({
        "Area_SqFt" : [Area_SqFt],
        "Rooms" : [Rooms],
        "Build_Year" : [Build_Year],
        "Location" : [Location],
        "Street_Type" : [Street_Type],
        "Furnishing" : [Furnishing],
        "Property_Type" : [Property_Type],
        "Has_Pool" : [Has_Pool]

    })

    #st.write("User input ")
    #st.write(input_data)

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=columns, 
                                          fill_value=0)

    #st.write("encoded input")
    #st.write(input_encoded)

    #numerical_columns = ["Area_SqFt", "Rooms", "Build_Year"]

    input_encoded[num_col] = scaler.transform(input_encoded[num_col])

    prediction = model.predict(input_encoded)

    st.success(f"Predicted House Price :{prediction[0]:,.2f}")

