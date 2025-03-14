import re
from transformers import pipeline
from kollygpt.GPT2Model import GPT2Model

class kollyGPTPlots:
    def __init__(self):
        self.tokenizer = GPT2Model().load_tokenizer()
        self.kollygpt = pipeline("text-generation", model="./gpt2-finetuned", tokenizer=self.tokenizer)
    
    def filter_english_words(text):
        words = text.split()
        filtered_words = [word for word in words if re.search(r'[A-Za-z]', word) and not re.search(r'[^A-Za-z.,!?]', word)]
        return ' '.join(filtered_words)

    def plotter(self, lead, genre):
        prompt = f"Prompt: Suggest a plot for a {genre} movie starring {lead} Response: "
        output = self.kollygpt(prompt, max_length=200, num_return_sequences=1, temperature=0.75, top_k=50, top_p=0.9)
        genPlot = output[0]["generated_text"].replace(prompt, "")
        return self.filter_english_words(genPlot)
