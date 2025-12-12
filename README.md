# Neural Language Model (PyTorch)

Train a GPT-style Transformer language model on Pride and Prejudice using PyTorch. The project includes tokenization, batching, model, training loop, evaluation (perplexity), plotting, checkpointing, and text generation.

## Requirements
- Python 3.9+
- PyTorch (CPU or CUDA)
- Matplotlib (for plots)

Install dependencies:

```
pip install torch matplotlib
```

## Dataset
- Provided at `dataset/Pride_and_Prejudice-Jane_Austen.txt`
- Tokenization is byte-level (`vocab_size=256`), no external tokenizers required

## Project Structure
- `data.py` — byte-level tokenizer and dataset batching
- `model.py` — GPT-style Transformer (causal attention, tied embeddings)
- `train.py` — training, evaluation, plotting, checkpointing, generation
- `config.py` — experiment presets (`underfit`, `overfit`, `bestfit`)
- `utils.py` — seeding, plotting, JSON logging, helpers
- `plots/` — saved train/val loss curves
- `runs/<exp>/metrics.json` — metrics per experiment
- `checkpoints/<exp>/best.pt` — best checkpoint by validation loss

## Quick Start
From the repository root:

```
python train.py --exp underfit
python train.py --exp overfit
python train.py --exp bestfit
```

These run the three required regimes and produce plots and metrics.

## Common Options
- `--epochs N` — override epochs
- `--batch_size B` — override batch size
- `--block_size L` — sequence length (context window)
- `--cpu` — force CPU
- `--amp` — enable mixed precision (CUDA)
- `--lowercase` — lowercase the dataset before training
- `--val_batches N` — validation batches per evaluation
- `--test_batches N` — test batches for final evaluation
- `--generate` — write a sample to `runs/<exp>/sample.txt`
- `--gen_tokens K` — number of tokens to generate
- `--temperature T` — sampling temperature
- `--top_k K` — top-k filtering during generation

## Examples
- Best-fit with AMP (GPU):
```
python train.py --exp bestfit --amp
```
- Best-fit on CPU with shorter run:
```
python train.py --exp bestfit --epochs 2 --cpu --val_batches 20 --test_batches 40
```
- Generate a sample from the best-fit run:
```
python train.py --exp bestfit --generate --gen_tokens 500 --temperature 0.9 --top_k 50
```

## Outputs
- Plot: `plots/<exp>.png` (train vs val loss)
- Metrics: `runs/<exp>/metrics.json` with `train_loss`, `val_loss`, `test_loss`, `val_perplexity`, `test_perplexity`
- Checkpoint: `checkpoints/<exp>/best.pt` (best validation loss)
- Sample: `runs/<exp>/sample.txt` (only if `--generate`)

## Reproducibility
- Fixed seed via `utils.set_seed(seed)`
- Deterministic cuDNN settings enabled when applicable
- Experiment configs defined in `config.py` for consistent runs

## Notes
- Underfit, overfit, and best-fit are configured to demonstrate model capacity and generalization differences.
- Byte-level tokenization ensures simple, reproducible preprocessing across the full text.
- For faster/better results, use a CUDA GPU (`--amp`) and consider increasing epochs.
