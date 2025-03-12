from transformers import pipeline
from kollygpt.GPT2Model import GPT2Model

class kollyGPTPlots:
    def __init__(self):
        self.tokenizer = GPT2Model().load_tokenizer()
        self.kollygpt = pipeline("text-generation", model="./gpt2-finetuned", tokenizer=self.tokenizer)

    def plotter(self, lead, genre):
        prompt = f"Prompt: Suggest a plot for a {genre} movie starring {lead} Response: "
        output = self.kollygpt(prompt, max_length=200, num_return_sequences=1)
        return output[0]["generated_text"].replace(prompt, "")
