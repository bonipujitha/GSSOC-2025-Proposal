# JAX & Flax LLM Examples - GSoC 2025 

This repository contains implementations for my Google Summer of Code 2025 proposal: "Create LLM Examples with JAX and Flax".

## Project Structure
```
├── notebooks/
│   ├── 01_basic_transformer.ipynb    # Basic transformer implementation
│   └── 02_training_example.ipynb     # Training demonstration
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py            # Transformer components
│   └── utils/
│       ├── __init__.py
│       └── training.py               # Training utilities
└── requirements.txt
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv jax_env
source jax_env/bin/activate  # Linux/Mac
# or
jax_env\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features Implemented
- Basic transformer block using JAX and Flax
- Simple training loop with optimization
- Example of GPU/TPU acceleration
- Educational comments explaining each component

## Usage
Check the `notebooks/` directory for interactive examples showing how to:
1. Build transformer components
2. Train on sample data
3. Optimize for performance

## License
MIT License
