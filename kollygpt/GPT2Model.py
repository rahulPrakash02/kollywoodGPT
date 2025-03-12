from kollygpt.GPT2Dataset import GPT2Dataset
from kollygpt.MovieData import MovieData
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

class GPT2Model:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def load_tokenizer(self):
        return self.tokenizer

    def train_model(self):
        movieData = MovieData()
        train_texts = movieData.load_train_texts()
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)

        train_dataset = GPT2Dataset(train_encodings)

        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned",
            evaluation_strategy="no",
            save_strategy="epoch",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            save_total_limit=2,
            report_to="none", 
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        trainer.train()

    def save_model(self):
        self.model.save_pretrained("./gpt2-finetuned")
        self.tokenizer.save_pretrained("./gpt2-finetuned")
    