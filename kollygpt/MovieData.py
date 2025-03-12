import pandas as pd

class MovieData:
    def __init__(self, file_name="./movie_data.csv"):
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name)
        self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

    def load_data(self):
        return self.data
    
    def load_leads(self):
        return self.data['Lead'].value_counts().index
    
    def print_data(self):
        print(self.data.head())
    
    def load_shape(self):
        return self.data.shape
    
    def load_train_texts(self):
        df = self.data
        df['prompt'] = "Suggest a plot for " + df['Genre'] + " movie starring " + df['Lead']
        df['response'] = df['Plot']
        df["text"] = "Prompt: " + df["prompt"] + " Response: " + df["response"]
        return df["text"].tolist()
        
        


