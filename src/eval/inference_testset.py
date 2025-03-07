import os
import json
import sys
import torch
import argparse
import random
from tqdm import tqdm,trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append('../model')
from modeling_gpt2_rope import GPT2LMHeadModelWithRoPE, GPT2ModelWithRoPE
from configuration_gpt2_rope import GPT2WithRoPEConfig
AutoConfig.register("gpt2-rope", GPT2WithRoPEConfig)
AutoModelForCausalLM.register(GPT2WithRoPEConfig, GPT2LMHeadModelWithRoPE)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--model_dir', type=str, default='gpt2-rope-5step-forward')
parser.add_argument('--order', type=str, default='forward')
parser.add_argument('--testset_path', type=str, default='../data/test.json')
parser.add_argument('--save_name', type=str, default='all_items.json')
parser.add_argument('--rewrite', type=bool, default=False)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
device = 'cuda:'+args.device
print('device: ',device)
batch_size=args.batch_size
print('batch_size: ',batch_size)
print('model_dir: ',args.model_dir)
print('order: ',args.order)
path='../LLaMA-Factory-main/saves/'+args.model_dir+'/full/sft'
print('testset_path: ',args.testset_path)
print('save_name: ',args.save_name)
print('rewrite: ',args.rewrite)
print('seed: ',args.seed)
random.seed(args.seed)

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer

def reorder_equations(data,order='reverse'):
    eqs = data['premise'].split("\n")
    if order=='forward':
        reorder_eqs="\n".join(eqs+[f"{data['target']}>>"])
    elif order=='reverse':
        reorder_eqs="\n".join(eqs[::-1]+[f"{data['target']}>>"])
    elif order=='random':
        random.shuffle(eqs)
        reorder_eqs="\n".join(eqs+[f"{data['target']}>>"])
    else:
        print('Please input correct order.')
        raise(NotImplementedError)
    return reorder_eqs

with open(args.testset_path) as f:
    lm_datasets=json.load(f)

for iter in tqdm(list(range(2000,1000000,2000))):
    if f'checkpoint-{str(iter)}' not in os.listdir(path):
        break
    if args.save_name in os.listdir(f'{path}/checkpoint-{str(iter)}'):
        print('Inference result exists!')
        if args.rewrite:
            print('Rewrite!')
        else:
            continue
    model, tokenizer = load_model(f'{path}/checkpoint-{str(iter)}')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    cnt = 0
    cnt_correct = 0
    ouf = []

    num_batches = (len(lm_datasets) + batch_size - 1) // batch_size

    for batch_idx in trange(num_batches):
        batch = lm_datasets[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        input_texts = [reorder_equations(item,args.order) for item in batch]
        inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
        
        generate_input = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 1,
            "repetition_penalty": 1.0,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id
        }
        
        generate_ids = model.generate(**generate_input)
        outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        
        for item, outp in zip(batch, outputs):
            item['order']=args.order
            item['model_output'] = outp
            try:
                parsed_ans = outp.split('>>')[-1]
            except:
                print(parsed_ans)
            if parsed_ans == item['gt']:
                cnt_correct += 1
            cnt += 1
            item['model_ans'] = parsed_ans
            ouf.append(item)

    print(f'checkpoint-{iter}:', cnt_correct / cnt)
    with open(f'{path}/checkpoint-{str(iter)}/{args.save_name}','w') as f:
        json.dump(ouf,f,ensure_ascii=False, indent=4)
