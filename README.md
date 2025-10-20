# Installation

## Install system dependencies (Linux)
```bash
# Install espeak-ng for text-to-speech functionality
sudo apt update
sudo apt install espeak-ng
```

## Install uv package manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create and activate virtual environment
```bash
# Create virtual environment using uv
uv venv -p 3.13

# Activate virtual environment
# Mac/Linux
source .venv/bin/activate

# Windows
.venv/Scripts/activate
```

## Install project dependencies
```bash
# Option 1: Install in editable mode
uv pip install -e .

# Option 2: Sync dependencies
uv sync
```

## Configuration
Create `.env` file to store token credentials:
```env
WANDB_API_KEY=your_wandb_api_key_here
```