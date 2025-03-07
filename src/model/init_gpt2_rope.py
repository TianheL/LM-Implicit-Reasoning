import os
from transformers import AutoConfig, AutoModelForCausalLM
from modeling_gpt2_rope import GPT2LMHeadModelWithRoPE
from transformers import AutoTokenizer
from configuration_gpt2_rope import GPT2WithRoPEConfig
from transformers import AutoConfig

os.environ['CUDA_VISIBLE_DEVICES']='0'

AutoConfig.register("gpt2-rope", GPT2WithRoPEConfig)
AutoModelForCausalLM.register(GPT2WithRoPEConfig, GPT2LMHeadModelWithRoPE)

config = AutoConfig.from_pretrained('.')
tokenizer=AutoTokenizer.from_pretrained('gpt2')
model=AutoModelForCausalLM.from_config(config)
model=model.eval()

assert(config._attn_implementation=='eager')
assert(model.config._attn_implementation=='eager')

model.save_pretrained('gpt2-rope')
tokenizer.save_pretrained('gpt2-rope')

print(model)