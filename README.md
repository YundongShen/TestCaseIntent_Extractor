# 🧪 Test Intent Extraction and Onboarding Document Generation System

An intelligent system that automatically extracts intent from test cases using DeepSeek LLM and generates onboarding documents. The system adopts a **five-layer architecture** for clear data processing and separation of concerns.

## 📋 Overview

- **Automated Intent Extraction**: Extract Activity, Goal, and Object from test code
- **Intelligent LLM Processing**: Use COT prompting to optimize and validate extraction results
- **Structured Document Generation**: Generate standardized onboarding documents from extracted intents
- **Tech Stack**: Python 3.8+, PyTorch, Transformers, DeepSeek 7B

## 🏗️ Project Structure

```
layers/                 # Five-layer processing pipeline
├── input/             # Layer 1: Data preprocessing
├── extract/           # Layer 2: Extract Activity, Goal, Object
├── intent/            # Layer 3: Validate and adjust intent
├── business/          # Layer 4: Document template and generation
└── output/            # Layer 5: Save final documents

model/                 # DeepSeek 7B model configuration and loading
testcases/             # Test case input files
onboarding_result/     # Generated onboarding documents
main.py                # Main pipeline entry point
```

## 🔄 Five-Layer Architecture

```
Raw Test Code
    ↓
[Layer 1] INPUT: Preprocess and normalize code
    ↓
[Layer 2] EXTRACT: Extract activities, goals, objects
    ↓
[Layer 3] INTENT: Validate and optimize extraction
    ↓
[Layer 4] BUSINESS: Load template and generate content
    ↓
[Layer 5] OUTPUT: Save Markdown document
    ↓
Final Onboarding Document
```

| Layer | Purpose | Key Component |
|-------|---------|----------------|
| 1 | Normalize test code | `preprocessor.py` |
| 2 | Extract intent elements | `*_extractor.py` |
| 3 | Validate & optimize | `validator.py`, `adjuster.py` |
| 4 | Business logic | `onboarding_generator.py` |
| 5 | Generate & save | `document_writer.py` |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- 16GB+ RAM (GPU recommended)
- Network connection (for model download)

### Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# or .venv\Scripts\activate # Windows

# Install dependencies
pip install -r requirements.txt

# Download DeepSeek model
python model/download_model.py
```

---

## 📝 Usage

### 1. Add Test Cases

Place test files in `testcases/` directory:
```bash
testcases/
├── admin-index-api.test.js
├── user-auth.test.js
└── payment-flow.test.js
```

Supported formats: `.test.js`, `.test.ts`, or any test code

### 2. Run Pipeline

```bash
python main.py
```

### 3. View Results

Generated documents in `onboarding_result/`:
```
onboarding_result/
├── onboarding_20260326_143022.md
├── onboarding_20260326_144534.md
└── onboarding_20260326_145056.md
```

---

## 🔧 Configuration

Edit `model/model_config.py` to adjust:
- `context_length`: Model context window size
- `device`: Use GPU/CPU for inference
- Model path and quantization settings

---

## 📚 Documentation

- **Module Details**: See `__init__.py` files in each layer
- **Example Test**: `testcases/admin-index-api.test.js`
- **Model Setup**: `model/model_config.py`

---

## 📄 License

MIT License
