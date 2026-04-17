# training

Training loops for the three stages of model development.

## Read in this order

1. **`pretrain.py`** - next-token prediction on unlabeled text. Cross-entropy loss over the shifted sequence. This is where a base model is made.
2. **`sft.py`** - supervised fine-tuning. Same loss as pretraining, but on curated instruction-response pairs. Loss is usually computed only on the response tokens, not the prompt.
3. **`grpo.py`** - Group Relative Policy Optimization. The RL method DeepSeek uses. Sample a group of responses per prompt, rank them by reward, update the policy toward the good ones. No value model needed (unlike PPO).

## Training as code vs. training runs

This repo is code-first. These files implement the training *loops* - correct loss, gradient step, scheduler, checkpointing, a 100-step sanity run on tiny data - but they are not meant to be run to convergence unless you want to. The point is that the loop is there, correct, and readable.
