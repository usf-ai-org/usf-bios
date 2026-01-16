# merge-lora
CUDA_VISIBLE_DEVICES=0 usf export \
    --adapters usf_bios/test_bert \
    --output_dir output/usf_test_bert_merged \
    --merge_lora true

# bnb quantize
CUDA_VISIBLE_DEVICES=0 usf export \
    --model output/usf_test_bert_merged \
    --output_dir output/usf_test_bert_bnb_int4 \
    --quant_bits 4 \
    --quant_method bnb

# infer
CUDA_VISIBLE_DEVICES=0 usf infer \
    --model output/usf_test_bert_bnb_int4
