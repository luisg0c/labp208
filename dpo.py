# DPO: alinhamento com preferencias humanas no dominio de musica eletronica

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

ADAPTER_PATH = "../labp107/adapter"
MODELO_BASE = "NousResearch/Llama-2-7b-hf"

BETA = 0.1
EPOCHS = 3
BATCH = 2
LR = 1e-5
MAX_LEN = 512

cfg_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

modelo = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    quantization_config=cfg_bnb,
    device_map="auto",
    is_trainable=True
)

tok = AutoTokenizer.from_pretrained(MODELO_BASE)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

ds = load_dataset("json", data_files="preferencias.jsonl", split="train")

args = DPOConfig(
    output_dir="adapter_dpo",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    optim="paged_adamw_32bit",
    beta=BETA,
    max_length=MAX_LEN,
    logging_steps=5,
    save_strategy="epoch",
)


if __name__ == "__main__":
    trainer = DPOTrainer(
        model=modelo,
        ref_model=None,
        processing_class=tok,
        train_dataset=ds,
        args=args,
    )

    trainer.train()
    trainer.model.save_pretrained("adapter_dpo")
    print("adapter salvo em adapter_dpo/")
