# CPU Training Guide for Vermeg Chatbot

## Quick Start for CPU Training

Since you're training on CPU, I've optimized everything for better performance:

### 1. Run the CPU Setup
```bash
setup_cpu_training.bat
```

### 2. Start Training
```bash
python finetune_vermeg_cpu.py
```

## What's Different for CPU Training

### Model Choice
- **GPU version**: Used Llama 3.1-8B (8 billion parameters)
- **CPU version**: Using DialoGPT-small (117 million parameters)
- **Why**: Smaller model = much faster training on CPU

### Training Optimizations
- **Batch size**: Larger (4 vs 1) since CPU handles batching differently
- **Sequence length**: Shorter (512 vs 1024) for faster processing
- **Epochs**: Fewer (2 vs 3) but higher learning rate
- **No mixed precision**: CPU doesn't support fp16

### Expected Performance

| Aspect | CPU Training | GPU Training |
|--------|-------------|-------------|
| **Setup time** | 5 minutes | 15+ minutes |
| **Training time** | 30-60 minutes | 10-20 minutes |
| **Model size** | 117M params | 8B params |
| **Quality** | Good for chatbot | Excellent |
| **Memory usage** | ~2GB RAM | ~12GB VRAM |

## Time Estimates

- **Setup**: 5 minutes
- **Model download**: 5 minutes  
- **Training**: 30-60 minutes (depends on your CPU)
- **Testing**: 2 minutes

**Total**: About 1 hour from start to finish

## CPU Requirements

### Minimum:
- 4 cores, 8GB RAM
- Will work but slowly

### Recommended:
- 8+ cores, 16GB+ RAM
- Much faster training

### Your 732 samples are perfect for CPU training!

## Alternative: Even Faster CPU Training

If you want even faster training, you can use an even smaller model:

Edit `finetune_vermeg_cpu.py` and change:
```python
model_name="distilgpt2"  # Only 82M parameters - very fast!
```

This will train in ~15-20 minutes but with slightly lower quality.

## After Training

The trained model will be saved as `./vermeg-chatbot-cpu/` and you can:

1. **Test it**: The script automatically tests with sample questions
2. **Deploy it**: Use it in your website/application
3. **Improve it**: Train longer or with more data if needed

## Quality Expectations

CPU-trained models with your 732 samples should be able to:
- ✅ Answer basic questions about Vermeg solutions
- ✅ Provide feature lists and benefits
- ✅ Handle simple comparisons
- ❌ May struggle with very complex reasoning
- ❌ May be less consistent than larger models

But for a business chatbot, this should work well!

## Ready to Start?

Just run:
```bash
setup_cpu_training.bat
python finetune_vermeg_cpu.py
```

The script will guide you through everything and show progress updates!
