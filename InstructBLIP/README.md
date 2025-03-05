# InstructBLIP 图像问答服务 API

## 简介

本项目是一个基于 Flask 框架搭建的图像问答服务 API，利用 InstructBLIP 模型实现根据输入的图像和文本提示进行问答。该 API 允许用户通过发送包含图像文件和文本提示的 POST 请求，获取 InstructBLIP 模型生成的答案。

## 环境要求

- **Python 环境**：确保已安装 Python，建议使用 Python 3.7 及以上版本。
- **依赖库**：
  - `torch`：深度学习框架，用于模型的计算。
  - `flask`：轻量级 Web 框架，用于搭建 API 服务。
  - `transformers`：用于加载和使用预训练的 Llava 模型和处理器。
  - `Pillow`：用于处理图像文件。

## 请求示例

```python
import requests

# 请求 URL
url = 'http://localhost:5000/predict'

# 图片文件路径
files = {'image': open('./demo.jpg', 'rb')}
data = {'prompt': 'What are these?'}

# 发送 POST 请求
response = requests.post(url, files=files, data=data)

# 检查响应状态码
if response.status_code == 200:
    # 解析 JSON 响应
    result_dict = response.json()
    result_text = result_dict.get('result')

    if result_text:
        answer = result_text[0]
        print(f"Question: {data['prompt']}")
        # Question: What are these?
        print(f"Answer: {answer}")
        # Answer: cars
else:
    print(f"Failed: {response.status_code}")
```
