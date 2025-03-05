import torch
import subprocess
import os
from flask import Flask, request, jsonify
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import logging

# 配置日志记录
logging.basicConfig(level=logging.ERROR)

# 模型名称
# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

print("Model:", model_name)
print("torch version:", torch.__version__)

custom_cache_dir = "/root/autodl-tmp/.hf-cache"
print("cache dir:", custom_cache_dir)

# 检查并创建缓存目录
if not os.path.exists(custom_cache_dir):
    os.makedirs(custom_cache_dir)

result = subprocess.run(
    'bash -c "source /etc/network_turbo && env | grep proxy"',
    shell=True,
    capture_output=True,
    text=True,
)
output = result.stdout
for line in output.splitlines():
    if "=" in line:
        var, value = line.split("=", 1)
        os.environ[var] = value

print("resource acceleration ✅")

app = Flask(__name__)

# 加载模型和处理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=custom_cache_dir,
)
processor = AutoProcessor.from_pretrained(
    model_name,
    cache_dir=custom_cache_dir,
)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 获取请求中的图像文件和文本提示
        file = request.files["image"]
        text_prompt = request.form.get("prompt", "What are these?")

        # 打开图像
        raw_image = Image.open(file.stream)

        # 构建对话模板
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image", "image": raw_image},  # 添加图像数据
                ],
            },
        ]

        # 应用对话模板生成提示
        prompt = processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)

        # 处理输入
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 检查 CUDA 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)

        # 生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        result = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return jsonify({"result": result})
    except Exception as e:
        logging.error("Error processing request: %s", str(e), exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
