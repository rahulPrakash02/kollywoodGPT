from .GPT2TrainLora import GPT2TrainLora
from .ModifiedGPT2Model import ModifiedGPT2Model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from peft import PeftModel

class GPT2Prediction:
    def __init__(self):
        self.model = PeftModel.from_pretrained(GPT2LMHeadModel.from_pretrained("gpt2"), "./lora_plot_generator_with_tokens_df")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<bos>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<eos>'})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_plot_dynamic_length_sliced_creative(self,lead, genre, initial_max_length=120, num_return_sequences=1,
                                                temperature=1.7,  # Higher for creativity
                                                top_p=0.99,     # Higher for creativity
                                                top_k=100,       # Higher for creativity
                                                no_repeat_ngram_size=3, # For non-repetition
                                                repetition_penalty=1.3, # For non-repetition
                                                min_length=30):
        
        prompt = f"<bos>Lead: {lead}\nGenre: {genre}\nPlot:"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=initial_max_length
        )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        output = self.model.to("cpu").generate(
            input_ids,
            attention_mask,
            max_length=initial_max_length,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
        )

        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt.replace('<bos>', "").strip(), "").strip()

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

        return final_plot.strip()