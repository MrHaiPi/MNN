import MNN.llm as llm
import MNN.cv as cv
import MNN.numpy as np
import sys
import PIL.Image as Image


def generate_example(model, prompt):
    prompt['text'] = model.apply_chat_template(prompt['text'])
    ids = model.tokenizer_encode(prompt)
    model.generate_init()
    logits = model.forward(ids)
    token = np.argmax(logits)
    model.context.current_token = token
    word = model.tokenizer_decode(token)
    print(word, end='', flush=True)
    for i in range(1024 * 64):
        logits = model.forward(token)
        token = np.argmax(logits)
        model.context.current_token = token
        if model.stoped():
            break
        word = ""
        try:
            word = model.tokenizer_decode(token)
        except:
            word = token
        print(word, end='', flush=True)


def response_example(model, prompt):
    # response stream
    model.response(prompt, True)
    vision_us = model.context.vision_us
    prefill_us = model.context.prefill_us
    decode_us = model.context.decode_us
    prompt_len = model.context.prompt_len
    decode_len = model.context.gen_seq_len
    pixels_mp = model.context.pixels_mp
    print('pixels : {}'.format(pixels_mp))
    print('vision time : {} ms'.format(vision_us / 1000.0))
    print('prefill speed : {} token/s'.format(prompt_len / (prefill_us / 1000000.0)))
    print('decode speed : {} token/s'.format(decode_len / (decode_us / 1000000.0)))


if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('usage: python vllm_example.py <path_to_model_config>')
    #     exit(1)

    # config_path = sys.argv[1]
    config_path = "/home/xiarui/nas/projects/MNN/output/model/Qwen3-VL-2B-Instruct-qb8/config.json"
    # create model
    model = llm.create(config_path)
    # load model
    model.load()


    img_path = '/home/xiarui/nas/projects/AIPhoto/Evaluate/raw/PhotoBench-v1/ai_camera_0a12e95c-51c7-4dd9-913a-9ba7f6512da8__1290AND1720.jpeg'
    # img_path = "/home/xiarui/nas/projects/AIPhoto/Evaluate/raw/PhotoBench-v1/ai_camera_1a843c8e-6c5f-4321-ae38-70da1a7904f4__1206AND1608.jpeg"
    img = cv.imread(img_path)

    with Image.open(img_path) as img_tmp:
        width, height = img_tmp.size
    print(f"Width: {width}, Height: {height}")
    
    prompt = {
        'text': '<img>{}</img>介绍一下这张图'.format(img_path),
#         "text": """<img>image_0</img>在此图像中找到最佳的摄影构图区域（着重考虑人像摄影的构图方式），并以JSON格式输出bbox坐标。输入图片的尺寸(宽*高)是420*420。
# **输出要求：**
# ```json
# {{
#   "bbox_2d": [x_min, y_min, x_max, y_max]
# }}
# ```
# """.strip(),
        'images': [
            {
                # 'data': img,
                # 'height': height,
                # 'width': width
            }
        ]
    }

    for _ in range(10):
        print("start to response")
        # response_example(model, prompt)
        generate_example(model, prompt)


