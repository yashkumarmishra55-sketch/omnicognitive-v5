# OmniCognitive V5 — Adaptive Information-Theoretic Logits Processor

An enterprise-grade, advanced custom `LogitsProcessor` for Hugging Face Transformers. It dynamically modulates large language model token distributions at run-time using streaming text entropy and varentropy.

## Novel Mechanisms (Patent-Pending)

* **Prompt-Conditioned Threshold Calibration:** Dynamically shifts processing thresholds based on initial prefill entropy.
* **Vocabulary-Partitioned Min-P:** Implements separate filtering tracks via `function_token_mask` to safeguard syntax tokens while allowing creative flexibility.
* **Chaos Memory Bank:** Tracks historical high-entropy states and automatically injects them back to forcefully break repetitive generation deadlocks.
* **Smooth EMA Chaos Feedback:** Continuously scales sampling operations smoothly using an Exponential Moving Average feedback matrix.

## Core Architectural Performance

| Preset Name | Base Threshold | Varentropy Weight (γ) | Top-K Range | Min-P Base | Chaos Streak Limit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `calm` | `1.6` | `0.10` | `2` to `24` | `0.05` | `2` |
| `balanced` | `1.2` | `0.15` | `4` to `40` | `0.08` | `3` |
| `creative` | `0.9` | `0.25` | `8` to `64` | `0.03` | `5` |

## Quick Start (Usage Example)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from omnicognitive.processor import OmniCognitiveV5

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Initialize your adaptive processor
prompt = "Your complex engineering prompt goes here..."
inputs = tokenizer(prompt, return_tensors="pt")

processor = OmniCognitiveV5(
    prompt_len=inputs.input_ids.shape[-1],
    preset="balanced"
)

# Apply during standard model generation
# (Ensure to integrate into your generation_config or logits_processor list)
```

## Global Technical Compliance Audit
All core algorithms successfully pass enterprise-grade functional parity tests including **Tensor Invariance**, **Non-Degenerate Pruning Safeties**, **Extreme Context Stress Testing (128k)**, and **Zero-Entropy Deadlock Verification**.

## License
* **Code:** Apache 2.0 License
* **Methods & Architecture:** Patent Pending
