from .ModifiedGPT2Model import ModifiedGPT2Model
from .MovieData import MovieData

class GPT2Embeddings:
    def __init__(self):
        self.tokenizer, _ = ModifiedGPT2Model().ModifiedGPT2()

    def format_prompt_from_df(self, row):
        bos_token = self.tokenizer.bos_token
        return f"{bos_token}Lead: {row['Lead']}\nGenre: {row['Genre']}\nPlot:"
    
    def return_embeddings(self):
        df = MovieData().load_data()
        eos_token = self.tokenizer.eos_token
        prompts_and_plots = df.apply(lambda row: self.format_prompt_from_df(row) + " " + row['Plot'] + f" {eos_token}", axis=1).tolist()
        train_encodings = self.tokenizer(prompts_and_plots, truncation=True, padding='max_length', return_tensors='pt')
        train_encodings = {key: val for key, val in train_encodings.items()}
        return train_encodings
    