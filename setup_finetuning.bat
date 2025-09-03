@echo off
echo ===============================================
echo Vermeg Chatbot Fine-tuning Setup Script
echo ===============================================

echo.
echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo.
echo Checking GPU availability...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing fine-tuning dependencies...
pip install -r finetuning_requirements.txt

echo.
echo Setup complete! 
echo.
echo To start fine-tuning, run:
echo python finetune_vermeg_chatbot.py
echo.
echo For a smaller model (if you have limited GPU memory):
echo Edit the script to use "microsoft/Phi-3-mini-4k-instruct"
echo.
echo For better performance (if you have 16GB+ VRAM):
echo Edit the script to use "meta-llama/Llama-3.1-8B-Instruct"
echo.
pause
