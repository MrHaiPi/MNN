cd /home/xiarui/nas/projects/MNN/transformers/llm/export

llmexport --path  /home/xiarui/nas/projects/LLaMA-Factory/output/qwen3vl_2b_thinking_lora_sft_20251101_gbs32_lr1e-4_ep20_lk32 --test "你好，请介绍一下你自己"


python llmexport.py --path /home/xiarui/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B \
--dst_path /home/xiarui/projects/MNN/output/model \
--export mnn


llmexport --path /home/xiarui/nas/projects/LLaMA-Factory/output/qwen3vl_2b_thinking_lora_sft_20251101_gbs32_lr1e-4_ep20_lk32 \
--dst_path /home/xiarui/nas/projects/MNN/output/model/qwen3vl_2b_thinking_lora_sft_20251101_gbs32_lr1e-4_ep20_lk32-qb4 \
--export mnn \
--quant_bit 4 

llmexport --path /home/xiarui/nas/projects/LLaMA-Factory/output/intern_vl3_1b_lora_sft_20251231_gbs32_lr1e-4_ep40_lk32_hf2cust \
--dst_path /home/xiarui/nas/projects/MNN/output/model/intern_vl3_1b_lora_sft_20251231_gbs32_lr1e-4_ep40_lk32_hf2cust-qb4 \
--export mnn \
--quant_bit 4

llmexport --path /home/xiarui/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct \
--dst_path /home/xiarui/nas/projects/MNN/output/model/Qwen2.5-VL-3B-Instruct-qb8 \
--export mnn \
--quant_bit 8

llmexport --path /home/xiarui/.cache/modelscope/hub/models/OpenGVLab/InternVL3-1B \
--dst_path /home/xiarui/nas/projects/MNN/output/model/InternVL3-1B-qb4 \
--export mnn \
--quant_bit 4 

llmexport --path /home/xiarui/.cache/modelscope/hub/models/OpenGVLab/InternVL3_5-1B-HF \
--dst_path /home/xiarui/nas/projects/MNN/output/model/InternVL3_5-1B-qb4 \
--export mnn \
--quant_bit 4

llmexport --path /home/xiarui/projects/LLaMA-Factory/output/qwen3vl_2b_instruct_lora_sft_20251031_gbs32_lr1e-4_ep80_lk32 \
--dst_path /home/xiarui/projects/MNN/output/model/qwen3vl_2b_instruct_lora_sft_20251031_gbs32_lr1e-4_ep80_lk32 \
--export mnn \
--quant_bit 8

