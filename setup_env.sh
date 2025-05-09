# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 is required but not found."
    echo "Please install Python 3.11 using:"
    echo "brew install python@3.11"
    exit 1
fi

# Create and activate virtual environment with Python 3.11
python3.11 -m venv consingan_env

# Activate virtual environment
source consingan_env/bin/activate

# Upgrade pip
pip3 install --upgrade pip

# Install PyTorch and other dependencies
pip3 install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy opencv-python matplotlib

# Clone ConSinGAN repository
git clone https://github.com/tohinz/ConSinGAN.git

# Copy training data
cp -r consingan_training/* ConSinGAN/

echo "Environment setup complete!"
echo "To activate the environment, run: source consingan_env/bin/activate"
echo "To start training, run:"
echo "cd ConSinGAN"
echo "python main.py --input_name training_image.png" 