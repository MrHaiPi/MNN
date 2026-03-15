import requests
import json

# 普通请求
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "mnn-vision-model",
        "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "描述一下这个图片"},
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                            }
                        },
                    ],
                }
            ]
    }
)
print(response.json())


