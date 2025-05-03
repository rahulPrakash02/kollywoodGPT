from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ModifiedGPT2Model:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

    def ModifiedGPT2(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '<bos>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({'eos_token': '<eos>'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        return self.tokenizer, self.model
    
    def save_tokenizer(self, output_adapter_path):
        self.tokenizer.save_pretrained(output_adapter_path)