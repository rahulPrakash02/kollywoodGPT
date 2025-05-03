import torch
from .ModifiedGPT2Model import ModifiedGPT2Model
from peft import LoraConfig, get_peft_model
from .GPT2Embeddings import GPT2Embeddings

class GPT2TrainLora:
    def __init__(self):
        _, self.model = ModifiedGPT2Model().ModifiedGPT2()
        self.loraConfig = LoraConfig(
            r=8,  # Rank of the LoRA matrices
            lora_alpha=32, # Scaling factor for LoRA weights
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"]
        )

    def create_dataloader(self, encodings, batch_size):
        for i in range(0, len(encodings['input_ids']), batch_size):
            yield {key: val[i:i+batch_size] for key, val in encodings.items()}

    def trained_model(self, output_adapter_path):
        self.model = get_peft_model(self.model, self.loraConfig)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        num_epochs = 1
        batch_size = 1
        dataloader = self.create_dataloader(GPT2Embeddings().return_embeddings(), batch_size)
        for epoch in range(num_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        self.model.save_pretrained(output_adapter_path)
        return self.model
