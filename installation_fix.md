# Manual Installation Guide for Vermeg Chatbot Fine-tuning

## Quick Fix for the Error

The error you encountered is because `flash-attn` requires CUDA development tools. Here's how to fix it:

### Step 1: Install Core Dependencies (No Optional Packages)
```bash
# Install PyTorch with CUDA support first
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
pip install transformers>=4.36.0
pip install datasets>=2.15.0
pip install accelerate>=0.24.0
pip install peft>=0.7.0

# Install utilities
pip install numpy pandas tqdm
pip install evaluate scikit-learn
```

### Step 2: Try bitsandbytes (Optional)
```bash
# This might work on Windows with CUDA
pip install bitsandbytes

# If it fails, that's OK - the training script will adapt
```

### Step 3: Verify Installation
```python
import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
```

### Step 4: Start Training
```bash
python finetune_vermeg_chatbot.py
```

## Alternative: CPU-Only Training (Slower but Works)

If you can't get CUDA working, you can still train on CPU:

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers datasets accelerate peft
pip install numpy pandas tqdm evaluate scikit-learn

# Run training (will be slower)
python finetune_vermeg_chatbot.py
```

## Cloud Alternatives (Recommended)

If local training is too slow or problematic:

### Google Colab (Free/Pro)
1. Upload your dataset files to Colab
2. Use the provided training script
3. Free tier: ~12 hours/day, T4 GPU
4. Pro tier: $10/month, better GPUs

### Kaggle Notebooks (Free)
1. Create a new notebook
2. Enable GPU acceleration
3. Upload dataset and run training
4. 30 hours/week free GPU time

### RunPod (Pay-per-use)
1. Rent GPU by the hour ($0.20-2.00/hour)
2. Pre-configured ML environments
3. Can pause/resume training

## Troubleshooting

### Common Issues:
1. **CUDA_HOME not set**: Install CUDA toolkit or use CPU training
2. **bitsandbytes fails**: Skip it, training will still work
3. **Out of memory**: Reduce batch size in the script
4. **Model loading fails**: Try a smaller model (Phi-3-mini)

### GPU Memory Requirements:
- **Phi-3-mini (3.8B)**: 6-8GB VRAM
- **Llama 3.2-3B**: 6-8GB VRAM  
- **Mistral-7B**: 12-16GB VRAM
- **Llama 3.1-8B**: 16-24GB VRAM

## Next Steps

Once you get the dependencies installed:

1. **Start with a small model** (Phi-3-mini) to test everything works
2. **Monitor GPU usage** during training
3. **Evaluate results** with the test questions
4. **Scale up** to larger models if needed

The cleaned dataset (732 samples) should give you good results even with smaller models!
