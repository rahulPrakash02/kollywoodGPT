from kollygpt.MovieData import MovieData
import streamlit as st
import os
from kollygpt.GPT2Model import GPT2Model
from kollygpt.kollyGPTPlots import kollyGPTPlots

model_path = "./gpt2-finetuned"

if not os.path.isdir(model_path):
    GPT2Model().train_model()
    GPT2Model().save_model()

st.title("Kollywood Plot Generator")

lead_choice = st.radio("Select your Lead:", MovieData().load_leads())

genre_choices = st.multiselect("Select your Genres:", ["Action", "Drama", "Romance", "Crime", "Thriller", "Horror", "Cop", "Spy", "Sport", "Comedy"])

if st.button("Generate Plot"):
    st.write(f"**Plot**\n {kollyGPTPlots().plotter(lead_choice, ', '.join(genre_choices))}")