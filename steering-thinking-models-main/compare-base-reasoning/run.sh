N_EXAMPLES=1
THINKING_TOKENS=1500
MAX_TOKENS=10000

python compare_reasoning.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS --max_tokens $MAX_TOKENS    

# python compare_reasoning.py --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

# python compare_reasoning.py --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

#python compare_reasoning.py --model "claude-3-opus-latest" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS

#python compare_reasoning.py --model "claude-3-7-sonnet-latest" --n_examples $N_EXAMPLES --thinking_tokens $THINKING_TOKENS
