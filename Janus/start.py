import torch
from transformers import AutoModelForCausalLM
from flask import Flask, request, jsonify
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import logging

# 配置日志记录
logging.basicConfig(level=logging.ERROR)

# 模型名称
model_name = "deepseek-ai/Janus-Pro-7B"

print("Model:", model_name)
print("torch version:", torch.__version__)

custom_cache_dir = "/root/autodl-tmp/.hf-cache"
print("cache dir:", custom_cache_dir)

app = Flask(__name__)

vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name,cache_dir=custom_cache_dir)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True,cache_dir=custom_cache_dir
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        prompt = request.form.get("prompt", "What are these?")
        images = request.form.get("image")
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [images],
            },
            {"role": "Assistant", "content": ""},
        ]

        # 加载图像并为输入做准备
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(vl_gpt.device)

        # 运行图像编码器以获取图像嵌入向量
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # 运行模型以获取响应
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        # print(f"{prepare_inputs['sft_format'][0]}", answer)
        return jsonify({"result": answer})
    except Exception as e:
        logging.error("Error processing request: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)