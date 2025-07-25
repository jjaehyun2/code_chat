import os
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import SFTTrainer

# CUDA 디바이스 설정 (필요에 따라 변경)
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 1) 경로 및 변수 설정
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"  # HuggingFace 허브 베이스 모델
ADAPTER_REPO = "jack0503/code-usage-model"    # 허브에 업로드한 어댑터 repo ID
DATA_PATH = "./pytorch_code_comment_pairs.json"  # JSON 데이터 경로 (Drive 기준)
SAVE_PATH = "./finetuned_model"                 # 학습 결과 저장 위치

# 2) JSON 데이터 로드
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
dataset = Dataset.from_list(data)

# 3) SFT 포맷 변환
def format_sft(example):
    return {
        "instruction": example["comment"],
        "input": "",
        "output": example["code"].strip()
    }
sft_dataset = dataset.map(format_sft)

def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}
train_dataset = sft_dataset.map(format_text)

# 4) 토크나이저 준비 (베이스 모델 기준)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    encoding = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )
    encoding["labels"] = encoding["input_ids"].copy()
    return encoding

tokenized_dataset = train_dataset.map(
    tokenize_function, batched=True, remove_columns=train_dataset.column_names
)

# 5) 모델 로드 및 어댑터 불러오기
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", use_cache=False)
base_model.gradient_checkpointing_enable()  # 메모리 절약용
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO, use_auth_token=True)

# 6) 학습 설정
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # 여유있게 시작
    learning_rate=2e-4,
    logging_steps=100,
    save_steps=500,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
)

print("SFT 파인튜닝 시작!")
trainer.train(resume_from_checkpoint=True)

print("저장 중...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"학습 완료! 저장위치: {SAVE_PATH}")
