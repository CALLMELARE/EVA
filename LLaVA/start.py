import torch
import subprocess
import os
import torch
from flask import Flask, request, jsonify
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

# 模型名称
model_name = "llava-hf/llava-1.5-7b-hf"

print("Model:", model_name)

print("torch version:", torch.__version__)

custom_cache_dir = "/root/autodl-tmp/.hf-cache"
print("cache dir:", custom_cache_dir)

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
model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir=custom_cache_dir,
).to(0)

processor = AutoProcessor.from_pretrained(model_name)


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
                    {"type": "image"},
                ],
            },
        ]

        # 应用对话模板生成提示
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # 处理输入
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(
            0, torch.float16
        )

        # 生成输出
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        # 解码输出
        result = processor.decode(output[0][2:], skip_special_tokens=True)

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
