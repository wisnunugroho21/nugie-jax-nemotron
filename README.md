# Nemotron 3 Nano – JAX Implementation

A **simple, minimalistic, and explainable** implementation of Nemotron 3 Nano in JAX/Flax NNX.

Nemotron 3 Nano is an efficient hybrid Mamba-Transformer model with Mixture-of-Experts (MoE), designed for agentic reasoning. This codebase prioritizes **clarity and educational value** over performance optimization, making it ideal for understanding how modern hybrid architectures work.

---

## 🎯 Project Goals

- **Explainability**: Every design choice is documented with clear comments.
- **Minimalism**: Unnecessary abstractions and optimizations are removed; only the essential concepts remain.
- **Reproducibility**: Small default dimensions allow full training on CPU/GPU without enterprise infrastructure.
- **Educational**: Serve as a reference for understanding Nemotron 3 Nano and hybrid LLM architectures.

---

## 🏗️ Architecture Overview

### Hybrid Stack Pattern

The model alternates between two types of mixer blocks:

- **Mamba 2 Blocks**: State-space models (SSMs) with linear-time complexity
- **Grouped-Query Attention (GQA)**: Efficient causal self-attention with fewer KV heads

Each mixer is followed by a **Sparse Mixture-of-Experts (MoE)** layer.

### Key Components

#### 1. **Mamba 2 Blocks** (`mamba_2.py`)
- State-space model layer with selective scanning (SSD algorithm)
- Processes sequences efficiently in O(n) time
- Uses input-dependent gating and a D skip connection for selective computation
- Chunked SSD algorithm keeps memory usage bounded for long sequences

#### 2. **Grouped-Query Attention** (`attention.py`)
- Causal masking (decoder-only) for language modeling
- Multiple query heads but shared KV heads (reduces parameters & memory)
- No positional embeddings, dropout, or bias on projections

#### 3. **Sparse Mixture-of-Experts** (`moe.py`)
- **Routed Experts**: Fine-grained expert specialization via granularity factors (DeepSeekMoE style)
- **Shared Experts**: Always-on experts for stable, universal computation
- **Sigmoid Gating**: Independent gate scores per expert (not softmax); top-k scores are renormalized
- **Squared-ReLU**: Stronger nonlinearity in expert FFNs
- **Bias-based Load Balancing**: Avoids auxiliary loss; expert biases are nudged with a simple sign rule after each step

#### 4. **Hybrid Model** (`nemotron.py`)
- Configurable layer pattern via `patterns` list (e.g., `mamba_moe` and `mamba_attention_moe` blocks)
- Pre-norm RMSNorm residual connections throughout
- Token embedding → N hybrid blocks → RMSNorm → LM head (untied weights)

---

## 📦 Installation

### Requirements
- Python 3.9+
- JAX (`jax[cpu]` or `jax[cuda]`)
- Flax
- Optax (for optimization)
- Orbax (for checkpointing)
- Datasets (for streaming FineWeb-Edu)
- Transformers (for Hugging Face tokenizers)

### Setup

```bash
# Clone or navigate to the project
cd nugie-jax-nemotron-3-nano

# Create and activate virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install "jax[cpu]" flax optax orbax-checkpoint datasets transformers
```

---

## 🚀 Usage

### Jupyter Notebook and Google Colab

Use the ready notebook at `notebooks/pretrain_nemotron.ipynb`.

Local Jupyter:

```bash
jupyter notebook notebooks/pretrain_nemotron.ipynb
```

Google Colab:

1. Open Colab and upload `notebooks/pretrain_nemotron.ipynb`.
2. Run all cells from top to bottom.

### Pretraining on FineWeb-Edu

`pretrained.py` implements the full pretraining workflow:

1. **Tokenization**: Loads the `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` tokenizer from Hugging Face.
2. **Dataset**: Streams text from `HuggingFaceFW/fineweb-edu` (no full download required).
3. **Training**: Next-token prediction with AdamW + linear warmup + cosine decay.
4. **Checkpointing**: Saves model weights via Orbax every `CHECKPOINT_EVERY` steps; resumes from latest checkpoint automatically.
5. **Evaluation**: Reports validation loss and perplexity after training.
6. **Interactive Chat**: Launches a terminal chat loop after training completes.

```bash
python pretrained.py
```

Key hyperparameters are constants at the top of `pretrained.py`:

```python
VOCAB_SIZE       = 131072   # Nemotron tokenizer vocabulary size
SEQ_LEN          = 256      # Tokens per training sample (must be divisible by CHUNK_SIZE)
CHUNK_SIZE       = 64       # Mamba SSD chunk size
BATCH_SIZE       = 2
LEARNING_RATE    = 3e-4
CHECKPOINT_EVERY = 200      # Save a checkpoint every N steps
MAX_TRAIN_STEPS  = 10000
WARMUP_STEPS     = 1000     # Linear warmup for the first N steps
VAL_STEPS        = 50       # Batches averaged for validation
MAX_GEN_TOKENS   = 200      # Max new tokens per chat response
MAX_CTX_LEN      = 512      # Rolling context window during generation
```

### Checkpointing

Model weights are saved using Orbax in `checkpoints/`. The training loop automatically resumes from the latest checkpoint if one exists:

```
checkpoints/
└── <step>/        # Orbax checkpoint directory per step
```

---

## 📂 Project Structure

```
nugie-jax-nemotron-3-nano/
├── pretrained.py       # Pretraining loop, evaluation, and interactive chat
├── nemotron.py         # Main model architecture (config + hybrid layer blocks)
├── attention.py        # Grouped-Query Attention (GQA) implementation
├── mamba_2.py          # Mamba 2 State-Space Model blocks (SSD algorithm)
├── moe.py              # Sparse Mixture-of-Experts implementation
├── notebooks/
│   └── pretrain_nemotron.ipynb  # Jupyter / Google Colab notebook
├── checkpoints/        # Orbax checkpoint directories (created at runtime)
├── LICENSE             # Apache 2.0
└── README.md           # This file
```

---

## 🔧 Configuration

Model architecture is configured via `NemotronConfig`. Three named presets are available through `NemotronConfig.from_preset()`:

| Preset | `d_model` | Layers | Notes |
|---|---|---|---|
| `tiny` *(default)* | 128 | 10 | Fits on any CPU; good for quick local tests |
| `kaggle` / `colab` | 256 | 13 | Medium size; fits a Kaggle/Colab GPU |
| `paper_close` | 2048 | 26 | Closest to the published Nemotron 3 Nano style |

```python
from nemotron import NemotronConfig, NemotronNanoBlock
from flax import nnx

config = NemotronConfig.from_preset("tiny")  # or "kaggle", "paper_close"
config.vocab_size = 131072                   # match your tokenizer

model = NemotronNanoBlock(rngs=nnx.Rngs(0), config=config)
```

Full list of `NemotronConfig` fields:

```python
NemotronConfig(
    vocab_size=1000,              # Vocabulary size (set from tokenizer)
    d_model=128,                  # Embedding / hidden dimension

    # Layer pattern: list of (block_type, repeats)
    # block_type ∈ {"mamba_moe", "mamba_attention_moe"}
    patterns=[("mamba_moe", 2), ("mamba_attention_moe", 1), ...],

    # Attention (GQA)
    num_attention_heads=4,        # Query heads
    num_kv_heads=1,               # KV heads (num_attention_heads % num_kv_heads == 0)
    attention_head_dim=32,        # num_attention_heads * attention_head_dim == d_model

    # Mamba-2 SSM
    mamba_d_state=64,             # SSM state dimension
    mamba_d_conv=4,               # Causal conv kernel width
    mamba_expand=2,               # Inner dim = mamba_expand * d_model
    mamba_headdim=64,             # Dimension per Mamba head
    mamba_ngroups=1,              # B/C groups (like GQA for Mamba)
    mamba_chunk_size=64,          # SSD chunk size (seq_len must be divisible)

    # Sparse MoE
    num_experts=4,                # Routed (base) expert count
    num_shared_experts=1,         # Always-on shared experts
    top_k=2,                      # Top-k routed experts per token
    expert_hidden_dim=256,        # Expert FFN hidden dimension
    granularity_factor=1,         # Splits each expert into finer sub-experts
    scale_top_k_with_granularity=True,  # Scale top_k by granularity_factor

    rms_norm_eps=1e-6,            # RMSNorm epsilon
)
```

`NemotronConfig.validate()` checks all shape constraints (e.g., `d_model == num_attention_heads * attention_head_dim`) and raises an `AssertionError` with a descriptive message if any constraint is violated.

---

## 📚 References

This implementation is inspired by:

1. **Nemotron 3 Nano Paper**: "Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"  
   [arXiv:2512.20848](https://arxiv.org/abs/2512.20848)

2. **Mamba 2 / SSD**: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Dao & Gu, 2024)  
   [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

3. **Mamba**: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"  
   [arXiv:2312.08636](https://arxiv.org/abs/2312.08636)

4. **Attention Is All You Need**: "Attention Is All You Need"  
   [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

5. **MoE Designs**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"  
   [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)

---

## 📝 License

Apache License 2.0 – See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

This is primarily an educational project. Feel free to:
- Open issues for bugs or clarifications
- Submit PRs with improvements or additional documentation
- Fork and adapt for your own experiments

---

## ⚠️ Status

**In Progress** – Core architecture is implemented and functional. Ongoing work includes:
- [ ] Performance benchmarking
- [ ] Longer sequence length testing
- [ ] Scaling to larger model sizes
- [ ] Advanced evaluation metrics

---

## 💡 Tips for Experimentation

1. **Start small**: Use the `tiny` preset to verify correctness locally before scaling up
2. **Monitor loss**: Watch for training instability in early steps; warmup helps
3. **Ablations**: Swap `mamba_attention_moe` blocks for `mamba_moe` to measure attention's contribution
4. **Text generation**: Use the interactive chat after training to qualitatively assess learned patterns
5. **Chunk size**: `SEQ_LEN` and `MAX_CTX_LEN` must both be divisible by `CHUNK_SIZE`

---

**Questions or suggestions?** Refer to inline code comments for detailed explanations of each component.
