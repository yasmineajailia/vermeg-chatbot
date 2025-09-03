@echo off
echo ===============================================
echo Vermeg Chatbot Fine-tuning Setup Script (Fixed)
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
echo Installing core dependencies first...
pip install numpy pandas tqdm

echo.
echo Checking current PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo.
echo The error suggests you have CPU-only PyTorch. Installing PyTorch with CUDA support...
echo This might take a few minutes...

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing core ML libraries...
pip install transformers>=4.36.0
pip install datasets>=2.15.0
pip install accelerate>=0.24.0
pip install peft>=0.7.0

echo.
echo Installing memory optimization (this might fail on some systems - that's OK)...
pip install bitsandbytes>=0.41.0

echo.
echo Installing evaluation libraries...
pip install evaluate>=0.4.0
pip install scikit-learn>=1.3.0

echo.
echo Verifying GPU setup...
python -c "import torch; print('=== GPU CHECK ==='); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('Device count:', torch.cuda.device_count()); print('Current device:', torch.cuda.current_device() if torch.cuda.is_available() else 'CPU only')"

echo.
echo ===============================================
echo Setup complete!
echo ===============================================
echo.
echo If you see "CUDA available: True" above, you're ready for GPU training!
echo If you see "CUDA available: False", you can still train but it will be much slower.
echo.
echo To start fine-tuning, run:
echo python finetune_vermeg_chatbot.py
echo.
echo Note: The script will automatically use the best available device (GPU or CPU)
echo.
pause
