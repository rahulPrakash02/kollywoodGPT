#Adjust Data
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
    
    def load_genres(self):
        #function to load genres instead of hardcoding list
        all_genres_string = ",".join(self.data['Genre'].astype(str))
        genre_list = all_genres_string.split(',')
        unique_genres_set = set(genre_list)
        return list(unique_genres_set)
        
        


