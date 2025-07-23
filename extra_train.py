import os
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 학습 GPU 번호 선택

# 이전에 CodeAlpaca로 파인튜닝된 모델 폴더
prev_ft_model_path = "./qwen-code-assistant-largemodel"
# HumanEval 추가 SFT 후 저장할 폴더
save_path = "./qwen-finetuned-v2"

# ----------------- HumanEval 데이터 준비 -----------------
def format_humaneval(example):
    instr = f"문제: 아래를 만족하는 파이썬 함수를 작성하세요.\n{example['prompt']}"
    return {
        "instruction": instr,
        "input": "",
        "output": example.get("canonical_solution", "").strip()
    }

# HumanEval 데이터셋 로드 및 포맷
human_eval = load_dataset("openai_humaneval")["test"]
human_eval_sft = human_eval.map(format_humaneval)
# 필요하다면, train-test split(여기선 전체 사용)

# 데이터를 SFT Trainer용 텍스트 형태로 변환
def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}

train_dataset = human_eval_sft.map(format_text)

# ----------------- 토크나이즈 -----------------
model = AutoModelForCausalLM.from_pretrained(prev_ft_model_path, device_map={"": 0})
tokenizer = AutoTokenizer.from_pretrained(prev_ft_model_path)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
        return_tensors=None
    )

tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)

# ----------------- SFT 파인튜닝 (HumanEval 추가 학습) -----------------
training_args = TrainingArguments(
    output_dir=save_path,
    num_train_epochs=1,              # HumanEval 데이터 적으니 epoch 1~2만 사용
    per_device_train_batch_size=2,   # VRAM에 맞게 조정 (필요시 1로 축소)
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=20,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None
)

print("HumanEval 추가 SFT 시작!")
trainer.train()
print("모델 저장 중...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"HumanEval 추가 SFT가 완료되었습니다. 저장 경로: {save_path}")
