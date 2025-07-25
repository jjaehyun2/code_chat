from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-3B-Instruct"

# 모델 다운로드
model = AutoModelForCausalLM.from_pretrained(model_id)

# 토크나이저 다운로드
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 모델과 토크나이저 로컬 저장 (원하는 경로로 지정 가능)
model.save_pretrained("./pretrained/qwen-3b-instruct")
tokenizer.save_pretrained("./pretrained/qwen-3b-instruct-tokenizer")
