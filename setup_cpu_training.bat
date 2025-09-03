@echo off
echo ===============================================
echo Vermeg Chatbot CPU Training Setup
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
echo Installing CPU-optimized PyTorch (no CUDA)...
pip install torch torchvision torchaudio

echo.
echo Installing core ML libraries...
pip install transformers>=4.36.0
pip install datasets>=2.15.0
pip install accelerate>=0.24.0
pip install peft>=0.7.0

echo.
echo Installing utilities...
pip install numpy pandas tqdm
pip install evaluate scikit-learn

echo.
echo Verifying installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('Device:', 'CPU'); print('Ready for CPU training!')"

echo.
echo ===============================================
echo CPU Setup Complete!
echo ===============================================
echo.
echo Your setup is optimized for CPU training.
echo Training will be slower than GPU but will work fine!
echo.
echo Estimated training time: 30-60 minutes
echo.
echo To start training, run:
echo python finetune_vermeg_cpu.py
echo.
echo The script uses a smaller, CPU-friendly model (DialoGPT-small)
echo that will still give you good results for your chatbot.
echo.
pause
