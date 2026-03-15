# mnn_openai_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import MNN.llm as llm
import uvicorn
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import MNN.cv as cv
import cv2
import io
import tempfile
import os

app = FastAPI(title="MNN LLM OpenAI-Compatible Server")

# 加载模型（全局单例）
config_path = "/home/xiarui/nas/projects/MNN/output/model/intern_vl3_1b_lora_sft_20251231_gbs32_lr1e-4_ep40_lk32_hf2cust-qb4/config.json"  # 替换为你的模型路径
model = llm.create(config_path)
# load model
model.load()

class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

def convert_openai_to_mnn_messages(openai_messages: List[Message]) -> List[Dict[str, Any]]:
    """
    将 OpenAI 格式的消息转换为 MNN LLM 支持的简单格式
    """
    mnn_messages = []
    
    image_index = 0
    for msg in openai_messages:
        if isinstance(msg.content, str):
            # 纯文本消息
            mnn_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        elif isinstance(msg.content, list):
            # 多模态消息（文本+图像）
            combined_content = ""
            images = []
            
            for content_item in msg.content:
                if content_item.type == "text" and content_item.text:
                    combined_content += content_item.text
                elif content_item.type == "image_url" and content_item.image_url:
                    # 处理 base64 图像数据
                    image_url = content_item.image_url.get("url", "")
                    
                    if image_url.startswith("data:"):
                        try:
                            # 自定义函数读取图片
                            image_stream = io.BytesIO(base64.b64decode(image_url.split(",", maxsplit=1)[1]))
                            cv2_image = Image.open(image_stream).convert("RGB")
                            tmp_file = "/home/xiarui/nas/projects/MNN/my_scripts/api/tmp.jpeg"
                            cv2_image.save(tmp_file)
                            # 添加到内容中
                            combined_content += f"<img>{tmp_file}</img>"

                            # 添加到图像列表
                            images.append({
                                # 'data': cv2_image, 
                                # 'height': height,
                                # 'width': width,
                            })
                            image_index += 1
                        except Exception as e:
                            print(f"Error processing image: {e}")
            
            mnn_messages.append({
                "role": msg.role,
                "content": combined_content
            })
    
    return mnn_messages, images

def generate_response(model, messages, max_new_tokens=1024, temperature=0.7):
    """
    使用 MNN LLM 生成响应
    """
    # 转换消息格式
    mnn_messages, images = convert_openai_to_mnn_messages(messages)
    
    # 使用 apply_chat_template 格式化对话
    formatted_prompt = {
        "text": model.apply_chat_template(mnn_messages),
        "images": images
    }
    
    ids = model.tokenizer_encode(formatted_prompt)

    # 初始化生成
    model.generate_init()
    
    # 处理第一个token
    logits = model.forward(ids)
    token = np.argmax(logits)
    model.context.current_token = token
    word = model.tokenizer_decode(token)
    
    # 收集生成的文本
    generated_text = word
    print(word, end='', flush=True)
    
    # 继续生成后续token
    for i in range(max_new_tokens - 1):
        logits = model.forward(token)
        token = np.argmax(logits)
        model.context.current_token = token
        
        # 检查是否停止生成
        if model.stoped():
            break
        
        try:
            word = model.tokenizer_decode(token)
        except:
            word = token
        generated_text += word
        print(word, end='', flush=True)
    
    return generated_text.strip()

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    try:
        # 调用 MNN LLM 生成响应
        response = generate_response(
            model=model,
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )
        
        # 计算token使用情况（近似值）
        mnn_messages = 0 #convert_openai_to_mnn_messages(request.messages)
        formatted_prompt = 0 #model.apply_chat_template(mnn_messages)
        prompt_tokens = 0 #len(model.tokenizer_encode(formatted_prompt))
        completion_tokens = 0 #len(model.tokenizer_encode(response))
        
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1730500000,
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    except Exception as e:
        return {
            "error": f"Failed to generate response: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
