from kollygpt.MovieData import MovieData
import streamlit as st
from kollygpt.GPT2Model import GPT2Model
from kollygpt.kollyGPTPlots import kollyGPTPlots

st.title("Kollywood Plot Generator")

lead_choice = st.radio("Select your Lead:", MovieData().load_leads())

genre_choices = st.multiselect("Select your Genres:", ["Action", "Drama", "Romance", "Crime", "Thriller", "Horror", "Cop", "Spy", "Sport", "Comedy"])

if st.button("Generate Plot"):
    st.write(f"**Plot**\n {kollyGPTPlots().plotter(lead_choice, ', '.join(genre_choices))}")