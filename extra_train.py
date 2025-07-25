from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 이전에 CodeAlpaca로 파인튜닝된 모델 폴더
prev_ft_model_path = "./qwen-code-assistant-largemodel"
# HumanEval 추가 SFT 후 저장할 폴더
save_path = "./qwen-finetuned-v2"

def format_humaneval(example):
    instr = f"문제: 아래를 만족하는 파이썬 함수를 작성하세요.\n{example['prompt']}"
    return {
        "instruction": instr,
        "input": "",
        "output": example.get("canonical_solution", "").strip()
    }

human_eval = load_dataset("openai_humaneval")["test"]
human_eval_sft = human_eval.map(format_humaneval)

def format_text(example):
    text = f"### Instruction:\n{example['instruction']}\n\n"
    if example.get("input", "").strip():
        text += f"### Input:\n{example['input']}\n\n"
    text += f"### Response:\n{example['output']}"
    return {"text": text}

train_dataset = human_eval_sft.map(format_text)

# 모델 로딩(명시적으로 cpu 지정)
model = AutoModelForCausalLM.from_pretrained("qwen-code-assistant-largemodel", device_map=None)
model = model.to("cpu")
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

training_args = TrainingArguments(
    output_dir=save_path,
    num_train_epochs=1,
    per_device_train_batch_size=1,   # CPU에서는 반드시 1로, 속도 개선 기대 어렵습니다
    learning_rate=2e-4,
    logging_steps=1,
    save_steps=20,
    fp16=False,                      # CPU에서는 반드시 False
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
