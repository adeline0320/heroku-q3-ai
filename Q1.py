import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

st.title('Question 1')
st.write("Here's our first attempt at using data to create a table: ")
st.write(pd.DataFrame({'first column': ["Money on-hand","Vacation duration","Hotel Star Rating","Tourist spots","One tourist spot","Food price","Transportation fees","Transport frequency"], 'second column': ["RM2000", "4 days", "<RM195 per night", "2 spot", "<RM 171","<RM95 per meal", "<RM187 per trip","4 trip per day"]}))

clear_data = pd.DataFrame(
    np.random.randn(20,3),
    columns = ['a', 'b', 'c'])

df = pd.read_csv('Vacation.csv')

print(df.to_string()) 
