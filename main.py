from model import Model
import streamlit as st

model = Model()

st.title('Text Editor App')
input_text = st.text_area("Type your text here:")
if st.button('Auto Correct Text'):
    edited_text = model.auto_correct(input_text.strip())
    st.text_area("Edited Text:", value=edited_text, height=300)
