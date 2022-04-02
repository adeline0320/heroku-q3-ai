import streamlit as st
import numpy as np
import pandas as pd

#st.set_page_config(layout = "wide")

st.title("Question 2")

st.write("State 1: ")
st.write(pd.DataFrame({'Vaccine Type' : ["Vac-A", "Vac-B", "Vac-C"], 'Age' : ["Age > 60", "Age = [35, 60]", "Age < 35"], 'Population' : [15000, 434890, 115900], "Day 1 - Day 8" : [1668, 1666, 1666], "Day 9" : [1656, 1672, 1672], "Day 10 - Day 49" : [0, 2500, 2500], "Day 50" : [0, 4100, 900], "Day 51 - Day 113" : [0, 5000, 0], "Day 114" : [0, 790, 0]}))
st.write(pd.DataFrame({'Vaccine Centre' : ["CR-1", "CR-2", "CR-3", "CR-4", "CR-5"], 'Max Vaccine Capacity per day' : [200, 500, 1000, 2500, 4000], 'Rental Cost per day (RM)' : [100, 250, 500, 800, 1200], "No. of centres (5000 vaccines)" : [0, 0, 0, 2, 0], "No. of centres (790 vaccines)" : [4, 0, 0, 0, 0]}))

st.write("State 2: ")
st.write(pd.DataFrame({'Vaccine Type' : ["Vac-A", "Vac-B", "Vac-C"], 'Age' : ["Age > 60", "Age = [35, 60]", "Age < 35"], 'Population' : [35234, 378860, 100450], "Day 1 - Day 10" : [3334, 3333, 3333], "Day 11" : [1894, 4053, 4053], "Day 12 - Day 23" : [0, 5000, 5000], "Day 24" : [0, 6933, 3067], "Day 25 - Day 51" : [0, 10000, 0], "Day 52" : [0, 4544, 0]}))
st.write(pd.DataFrame({'Vaccine Centre' : ["CR-1", "CR-2", "CR-3", "CR-4", "CR-5"], 'Max Vaccine Capacity per day' : [200, 500, 1000, 2500, 4000], 'Rental Cost per day (RM)' : [100, 250, 500, 800, 1200], "No. of centres (10000 vaccines)" : [0, 0, 0, 4, 0], "No. of centres (4544 vaccines)" : [3, 0, 0, 0, 1]}))

st.write("State 3: ")
st.write(pd.DataFrame({'Vaccine Type' : ["Vac-A", "Vac-B", "Vac-C"], 'Age' : ["Age > 60", "Age = [35, 60]", "Age < 35"], 'Population' : [22318, 643320, 223400], "Day 1 - Day 8" : [2500, 2500, 2500], "Day 9" : [2318, 2591, 2591], "Day 10 - Day 62" : [0, 3750, 3750], "Day 63" : [0, 5441, 2059], "Day 64 - Day 118" : [0, 7500, 0], "Day 119" : [0, 4038, 0]}))
st.write(pd.DataFrame({'Vaccine Centre' : ["CR-1", "CR-2", "CR-3", "CR-4", "CR-5"], 'Max Vaccine Capacity per day' : [200, 500, 1000, 2500, 4000], 'Rental Cost per day (RM)' : [100, 250, 500, 800, 1200], "No. of centres (7500 vaccines)" : [0, 0, 0, 3, 0], "No. of centres (4038 vaccines)" : [1, 0, 0, 0, 1]}))

st.write("State 4: ")
st.write(pd.DataFrame({'Vaccine Type' : ["Vac-A", "Vac-B", "Vac-C"], 'Age' : ["Age > 60", "Age = [35, 60]", "Age < 35"], 'Population' : [23893, 859900, 269300], "Day 1 - Day 8" : [2834, 2833, 2833], "Day 9" : [1221, 3639, 3640], "Day 10 - Day 66" : [0, 4250, 4250], "Day 67" : [0, 7754, 746], "Day 68 - Day 135" : [0, 8500, 0], "Day 136" : [0, 5593, 0]}))
st.write(pd.DataFrame({'Vaccine Centre' : ["CR-1", "CR-2", "CR-3", "CR-4", "CR-5"], 'Max Vaccine Capacity per day' : [200, 500, 1000, 2500, 4000], 'Rental Cost per day (RM)' : [100, 250, 500, 800, 1200], "No. of centres (8500 vaccines)" : [0, 0, 1, 3, 0], "No. of centres (5593 vaccines)" : [3, 0, 0, 2, 0]}))

st.write("State 5: ")
st.write(pd.DataFrame({'Vaccine Type' : ["Vac-A", "Vac-B", "Vac-C"], 'Age' : ["Age > 60", "Age = [35, 60]", "Age < 35"], 'Population' : [19284, 450500, 221100], "Day 1 - Day 6" : [3168, 3166, 3166], "Day 7" : [276, 4612, 4612], "Day 8 - Day 48" : [0, 4750, 4750], "Day 49" : [0, 6758, 2742], "Day 50 - Day 72" : [0, 9500, 0], "Day 73" : [0, 6884, 0]}))
st.write(pd.DataFrame({'Vaccine Centre' : ["CR-1", "CR-2", "CR-3", "CR-4", "CR-5"], 'Max Vaccine Capacity per day' : [200, 500, 1000, 2500, 4000], 'Rental Cost per day (RM)' : [100, 250, 500, 800, 1200], "No. of centres (9500 vaccines)" : [0, 1, 0, 2, 1], "No. of centres (6884 vaccines)" : [2, 0, 0, 1, 1]}))

# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)


# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns = ['States', 'Duration (days)', 'Cost (RM)']
# )

# st.line_chart(chart_data)