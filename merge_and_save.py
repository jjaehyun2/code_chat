from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. adapter(파인튜닝 모델), base model 경로 지정
adapter_path = "./qwen-code-assistant-final"
base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"   # (원본 base 모델명)

# 2. base 모델 로드
model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map="cpu")

# 3. PEFT(LoRA) 파인튜닝 모델 로드
model = PeftModel.from_pretrained(model, adapter_path)

# 4. merge and unload
model = model.merge_and_unload()

# 5. 새 폴더에 통합 모델 저장
save_path = "./qwen-code-assistant-merged"
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.save_pretrained(save_path)
