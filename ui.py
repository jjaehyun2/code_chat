import os
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# 모델 경로
model_dir = "./qwen-code-assistant-largemodel"  # 또는 'final' 경로
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", torch_dtype='auto')

def chat(instruction, input_text=""):
    prompt = f"### Instruction:\n{instruction}\n\n"
    if input_text.strip():
        prompt += f"### Input:\n{input_text}\n\n"
    prompt += "### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:\n")[1].strip() if "### Response:" in response else response

demo = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="Instruction"),
        gr.Textbox(label="Input ")
    ],
    outputs="text",
    title="Prototype Code Assistant"
)

demo.launch(share=True)  # share=True면 외부 접속도 가능
