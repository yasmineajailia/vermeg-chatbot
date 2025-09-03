# Vermeg Chatbot Fine-tuning Guide

## Open-Source Model Recommendations

### 1. **Recommended Models by Size & Use Case**

#### Small Models (Good for local deployment, fast inference)
- **Phi-3-mini (3.8B)** - Microsoft's efficient model, excellent for Q&A
- **Llama 3.2-3B** - Meta's latest small model, great performance
- **Qwen2.5-3B** - Alibaba's model, strong for customer support

#### Medium Models (Balanced performance & resources)
- **Llama 3.1-8B** - Excellent for customer service chatbots
- **Mistral-7B-v0.3** - Great instruction following
- **Qwen2.5-7B** - Strong multilingual support

#### Large Models (Best performance, higher resource requirements)
- **Llama 3.1-70B** - Enterprise-grade performance
- **Qwen2.5-72B** - Excellent reasoning capabilities

### 2. **Recommended Choice for Your Use Case**
**Llama 3.1-8B-Instruct** - Best balance of performance and efficiency for a business chatbot

## Fine-tuning Approaches

### Option 1: Using Hugging Face (Easiest)
```python
# Install requirements
pip install transformers datasets peft accelerate bitsandbytes

# Fine-tuning script
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True  # Use 4-bit quantization to save memory
)

# LoRA configuration for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load and prepare dataset
dataset = load_dataset('json', data_files='vermeg_dataset_instruction.json')
```

### Option 2: Using Unsloth (Faster, Memory Efficient)
```python
# Install Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

from unsloth import FastLanguageModel
import torch

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8b-instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

### Option 3: Using Axolotl (Professional Tool)
```yaml
# config.yml for Axolotl
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer

load_in_8bit: false
load_in_4bit: true
strict: false

datasets:
  - path: vermeg_dataset_instruction.json
    type: alpaca

dataset_prepared_path: ./prepared
val_set_size: 0.1
output_dir: ./vermeg-llama-lora

adapter: lora
lora_model_dir:
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true
lora_fan_in_fan_out:

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

wandb_project: vermeg-chatbot
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

train_on_inputs: false
group_by_length: false
bf16: auto
fp16:
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:

logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 4
eval_table_size:
eval_max_new_tokens: 128
saves_per_epoch: 1
debug:
deepspeed:
weight_decay: 0.0
fsdp:
fsdp_config:
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 3080/4070 (12GB VRAM) or better
- **RAM**: 32GB system RAM
- **Storage**: 100GB free space

### Recommended Setup
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or A100
- **RAM**: 64GB system RAM
- **Storage**: 500GB SSD

### Cloud Options
- **Google Colab Pro+**: $50/month, good for experimentation
- **Paperspace Gradient**: Pay-per-use, various GPU options
- **RunPod**: Affordable GPU rental
- **AWS/Azure/GCP**: Enterprise options

## Training Parameters

### Recommended Settings for Your Dataset (732 samples)
```python
training_args = TrainingArguments(
    output_dir="./vermeg-llama-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
)
```

## Post-Training Steps

1. **Model Evaluation**: Test with validation questions
2. **Quantization**: Convert to GGUF format for deployment
3. **Deployment**: Use Ollama, vLLM, or similar for serving
4. **Integration**: Connect to your website/application

## Estimated Costs & Time

### Training Time
- **Local (RTX 4090)**: 2-4 hours
- **Cloud (A100)**: 1-2 hours
- **Colab Pro+**: 3-6 hours

### Costs (Cloud)
- **Training**: $10-30 per experiment
- **Inference**: $0.001-0.01 per query

## Next Steps

1. Choose your model size based on deployment needs
2. Set up your training environment
3. Run fine-tuning with the cleaned dataset
4. Evaluate and iterate
5. Deploy for production use

Would you like me to create specific scripts for any of these approaches?
