# 原始说明文档
https://github.com/alibaba/MNN/blob/master/docs/transformers/llm.md

https://mnn-docs.readthedocs.io/en/latest/transformers/llm.html#id8

# 编译
cd MNN
mkdir build && cd build
cmake .. -DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DLLM_SUPPORT_VISION=true -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true
make -j8

# 使用config.json
## 交互式聊天
./llm_demo model_dir/config.json
## 针对prompt中的每行进行回复
./llm_demo model_dir/config.json prompt.txt

# 不使用config.json, 使用默认配置
## 交互式聊天
./llm_demo model_dir/llm.mnn
## 针对prompt中的每行进行回复
./llm_demo model_dir/llm.mnn prompt.txt

<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容
# 指定图片大小
<img><hw>280, 420</hw>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>介绍一下图片里的内容

<audio>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav</audio>介绍一下音频里的内容


# 聊天运行
./llm_demo /home/xiarui/projects/MNN/output/model/Qwen3-VL-2B-Instruct-qb4/config.json 

# 批量运行
./llm_demo /home/xiarui/nas/projects/MNN/output/model/intern_vl3_1b_lora_sft_20251231_gbs32_lr1e-4_ep40_lk32_cp800_hf2cust-qb8/config.json /tmp/tmpy7h9dz95.txt
