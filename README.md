# PL-BERT-ID

## Prerequisites

- Python 3.13
- Linux/macOS/Windows
- espeak-ng (for text-to-speech functionality)

## Installation

### 1. Install System Dependencies

**Linux:**
```bash
sudo apt update
sudo apt install espeak-ng
```

**macOS:**
```bash
brew install espeak-ng
```

**Windows:**
Download and install from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)

### 2. Install UV Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Create Virtual Environment

```bash
# Create virtual environment using uv
uv venv -p 3.13

# Activate virtual environment
# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install Project Dependencies

```bash
# Install in editable mode (recommended for development)
uv pip install -e .

# Or sync dependencies from lock file
uv sync
```

## Dataset Setup

### Download Wikipedia Indonesian Dataset

1. Create a `wikipedia.id` folder in the project root:
   ```bash
   mkdir -p wikipedia.id
   ```

2. Download the parquet file from HuggingFace:
   - Visit: https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id
   - Download all parquet files
   - Save them in the `wikipedia.id` folder

## Usage

### Training Pipeline

1. **Preprocess the data:**
   ```bash
   python preprocessing.py
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```