# Import
import streamlit as st

st.title('Forecast for sales')

st.header('Management inventory')

# st.dataframe(my_dataframe)


# Checkbox to display something
if st.checkbox('Show 1'):
    st.write('''
        Screen 1 : Inventory days of supply
        ''')

if st.checkbox('Show 2'):
    st.write('''
        Screen 2 : Top 10 of sales, invetories
        ''')

    st.write('''
        Screen 2 : Needed product (Prod, Alert, Nb)
        ''')

if st.checkbox('Show 3'):
    st.write('''
        Screen 3 : Inventory Trend
        ''')

    st.write('''
        Screen 3 : Inventory Efficient (Lines predict, Lines real)
        ''')

if st.checkbox('Show 4'):
    st.write('''
        Screen 4 : Available stock by departement
        ''')

    # days with sliders, textarea
    st.write('''
        Screen 4 : Expire stock within 10 days
        ''')

if st.checkbox('Show Map'):
    st.write('''
        Screen Map : with Folium
        ''')

    # search latitude, longitude
    # coordinates of city, state
    # display details of stores when hover with mousepad
