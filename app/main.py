import streamlit as st

# App title
st.title("My First Streamlit App")

# Text
st.write("Hello! This is a simple Streamlit app 🤖")

# Input
name = st.text_input("What's your name?")

# Button + output
if st.button("Say hi"):
    if name:
        st.success(f"Nice to meet you, {name}!")
    else:
        st.warning("Please type your name first 🙂")
