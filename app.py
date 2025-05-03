from kollygpt.MovieData import MovieData
import streamlit as st
import os
from peft import PeftConfig
from kollygpt.GPT2TrainLora import GPT2TrainLora
from kollygpt.ModifiedGPT2Model import ModifiedGPT2Model

model_path = "./lora_plot_generator_with_tokens_df"

if not os.path.isdir(model_path):
    GPT2TrainLora().trained_model(model_path)
    ModifiedGPT2Model().save_tokenizer(model_path)

tokenizer, model = ModifiedGPT2Model().ModifiedGPT2()
peft_config = PeftConfig.from_pretrained("./lora_plot_generator_with_tokens_df")
bos_token = "<bos>"

def generate_plot_dynamic_length_sliced_creative(lead, genre, initial_max_length=120, num_return_sequences=1,
                                                temperature=1.7,  # Higher for creativity
                                                top_p=0.99,     # Higher for creativity
                                                top_k=100,       # Higher for creativity
                                                no_repeat_ngram_size=3, # For non-repetition
                                                repetition_penalty=1.3, # For non-repetition
                                                min_length=30):
    prompt = f"{bos_token}Lead: {lead}\nGenre: {genre}\nPlot:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cpu")

    output = model.generate(
        input_ids,
        max_length=initial_max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt.replace(bos_token, "").strip(), "").strip()

    # Split into sentences and take the first two complete ones
    sentences = generated_text.split('.')
    meaningful_plot = ""
    sentence_count = 0
    for sentence in sentences:
        cleaned_sentence = sentence.strip()
        if cleaned_sentence:
            meaningful_plot += cleaned_sentence + "."
            sentence_count += 1
            if sentence_count >= 2 and len(meaningful_plot.split()) >= min_length:
                break

    final_plot = meaningful_plot.strip()
    if final_plot.endswith('.'):
        final_plot = final_plot
    elif '.' in final_plot:
        final_plot = final_plot.rsplit('.', 1)[0].strip() + "."
    # else: keep it as is

    return final_plot.strip().replace(bos_token, "")

st.title("Kollywood Plot Generator")

lead_choice = st.radio("Select your Lead:", MovieData().load_leads())

genre_choices = st.multiselect("Select your Genres:", MovieData().load_genres())

if st.button("Generate Plot"):
    st.write(f"**Plot**\n {generate_plot_dynamic_length_sliced_creative(lead_choice, (', '.join(genre_choices)))}")