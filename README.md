# MoE-PBA

## Installation

```bash
bash scripts/install.sh
source .venv/bin/activate
```

## Example

Prune 25% of OLMoE's routed experts with MoE-PBA and run code evaluations:

```bash
bash scripts/run_eval.sh \
  --model allenai/OLMoE-1B-7B-0125-Instruct \
  --ratio 0.25 \
  --domain code \
  --gpus 0
```
