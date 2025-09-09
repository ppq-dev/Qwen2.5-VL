# Qwen2.5-VL

[![GitHub Repo stars](https://img.shields.io/github/stars/QwenLM/Qwen2.5-VL?style=social)](https://github.com/QwenLM/Qwen2.5-VL/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/QwenLM/Qwen2.5-VL?style=social)](https://github.com/QwenLM/Qwen2.5-VL/network/members)
[![GitHub issues](https://img.shields.io/github/issues/QwenLM/Qwen2.5-VL)](https://github.com/QwenLM/Qwen2.5-VL/issues)
[![GitHub license](https://img.shields.io/github/license/QwenLM/Qwen2.5-VL)](https://github.com/QwenLM/Qwen2.5-VL/blob/main/LICENSE)

Qwen2.5-VL is the multimodal version of the Qwen2.5 series models, supporting both images and text as inputs. It is built upon Qwen2.5 and incorporates visual capabilities.

## Model Summary

- **Developed by:** [Qwen Team](https://qwenlm.github.io/), Alibaba Group
- **Model type:** Transformer-based multimodal language model
- **Languages:** Multilingual
- **License:** [Qwen2.5 License](https://github.com/QwenLM/Qwen2.5/blob/main/LICENSE)
- **Maximum Sequence Length:** 131,072 tokens

## Model Sizes

| Model Name | Description |
|------------|-------------|
| Qwen2.5-VL-2B | 2 Billion parameters |
| Qwen2.5-VL-7B | 7 Billion parameters |
| Qwen2.5-VL-72B | 72 Billion parameters |

## Quick Start

### Installation

```bash
pip install transformers accelerate
```

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare your multimodal input (text + image)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
```

## Performance

Qwen2.5-VL demonstrates strong performance across various multimodal benchmarks including:
- MMMU
- MMBench
- SEED-Bench
- DocVQA
- TextVQA

## Training Details

Qwen2.5-VL is trained on a large-scale multimodal dataset including:
- High-quality image-text pairs
- Interleaved image-text documents
- Instructional data

## Safety & Alignment

The model undergoes rigorous safety training and alignment to ensure responsible AI deployment.

## Citation

If you find Qwen2.5-VL useful, please cite our work:

```bibtex
@misc{qwen2.5-vl,
  title={Qwen2.5-VL: A Multimodal Language Model Supporting Vision and Language},
  author={Qwen Team},
  year={2024},
  howpublished={\url{https://github.com/QwenLM/Qwen2.5-VL}}
}
```

## Related Projects

1. related project [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2)
2. related project [Aria](https://github.com/rhymes-ai/Aria)
3. related project [Kimi-VL](https://github.com/MoonshotAI/Kimi-VL)