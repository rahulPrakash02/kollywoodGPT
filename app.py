from kollygpt.MovieData import MovieData
import streamlit as st
import os
from kollygpt.GPT2Prediction import GPT2Prediction
from kollygpt.GPT2TrainLora import GPT2TrainLora
from kollygpt.ModifiedGPT2Model import ModifiedGPT2Model

model_path = "./lora_plot_generator_with_tokens_df"

#GPT2TrainLora().trained_model(model_path)
#ModifiedGPT2Model().save_tokenizer(model_path)

st.title("Kollywood Plot Generator")

lead_choice = st.radio("Select your Lead:", MovieData().load_leads())

genre_choices = st.multiselect("Select your Genres:", MovieData().load_genres())

if st.button("Generate Plot"):
    st.write(f"**Plot**\n {GPT2Prediction().generate_plot_dynamic_length_sliced_creative(lead_choice, (', '.join(genre_choices)))}")