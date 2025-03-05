from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
import os
from flask import Flask, request, jsonify
from PIL import Image
import requests
import logging

# 配置日志记录
logging.basicConfig(level=logging.ERROR)

# 模型名称
model_name = "Salesforce/instructblip-vicuna-7b"

print("Model:", model_name)
print("torch version:", torch.__version__)

custom_cache_dir = "/root/autodl-tmp/.hf-cache"
print("cache dir:", custom_cache_dir)

app = Flask(__name__)


# 加载模型和处理器
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
)
processor = InstructBlipProcessor.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取请求中的图像文件和文本提示
        file = request.files["image"]
        text_prompt = request.form.get("prompt", "What are these?")

        # 打开图像
        image = Image.open(file.stream).convert("RGB")
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(
            device
        )

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(result)
        return jsonify({"result": result})
    except Exception as e:
        logging.error("Error processing request: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
