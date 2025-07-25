import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # GPU 설정

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. 데이터셋 로드 및 전처리
dataset = load_dataset("sahil2801/CodeAlpaca-20k")
train_val = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val["train"]
val_dataset = train_val["test"]

def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}



model_id = "Qwen/Qwen2.5-3B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
# ⭐️ 토크나이즈 함수 적용!
def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,  # 원하는 길이로
        return_tensors="pt"
    )
    
train_dataset = train_dataset.map(format_text)
val_dataset = val_dataset.map(format_text)

# 이때 반드시 batched=True, remove_columns 적용!
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)



model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': 0},
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./qwen-code-assistant",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    fp16=True,
)

# **SFTTrainer 직접 호출**
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

print("Starting training...")
trainer.train()
print("Saving model...")
trainer.save_model("./qwen-code-assistant-largemodel")
print("Training complete!")