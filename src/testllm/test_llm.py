# encoding=utf-8
import random
from tqdm import tqdm  
import argparse
import json
import os
from openai import OpenAI
import time
from collections import Counter

random.seed(42)

client = OpenAI(
    api_key = os.environ['OPENAI_API_KEY'],
    base_url = os.environ['OPENAI_API_BASE']
)

parser = argparse.ArgumentParser()
parser.add_argument('--step', type=int, default=3)
parser.add_argument('--model', type=str, default='gpt-4o-2024-08-06')
parser.add_argument('--order', type=str, default='forward')
args = parser.parse_args()
print('model: ',args.model)
print('step: ',args.step)
print('order: ',args.order)
print()

def request_model(prompt):
    for _ in range(30):
        try:
            result = client.chat.completions.create(
                temperature=0,
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False
            )
            res = result.choices[0].message.content
            break
        except Exception as e:
            print(e)
            res = 'Request failed!'
            time.sleep(random.randint(1,2))
    return res

def reorder_equations(equations):
    eqs = equations.split("\n")
    if args.order=='forward':
        reorder_eqs = '\n'.join(eqs)
    elif args.order=='reverse':
        reorder_eqs = '\n'.join(eqs[::-1])
    elif args.order=='random':
        reorder_eqs = random.sample(eqs,len(eqs))
        reorder_eqs = '\n'.join(reorder_eqs)
    # specify your order
    elif args.order.isdigit():
        if len(args.order)!=len(eqs):
            print('Error: length not match!')
            print()
        if set(args.order)!=set(str(i) for i in range(len(eqs))):
            print('Error: miss index!')
            print()
        assert len(args.order)==len(eqs) and set(args.order)==set(str(i) for i in range(len(eqs)))
        reorder_eqs = []
        for i in args.order:
            reorder_eqs.append(eqs[int(i)])
        reorder_eqs = '\n'.join(reorder_eqs)
    else:
        raise NotImplementedError
    return reorder_eqs


if __name__ == '__main__':
    with open(f'./data/testdata_step{str(args.step)}.json') as f:
        test_data=json.load(f)
    input_str = test_data[0]['premise']
    output_str = reorder_equations(input_str)
    print('===original===')
    print(input_str)
    print('===reordered===')
    print(output_str)
    print()

    for item in tqdm(test_data):
        item['prompt']=reorder_equations(item['premise'])+f'\nWhat is the value of {item["target"]}? You must answer directly. Only output the final result. Begin your answer with "{item["target"]} = xx".'
        if 'model_output' not in item:
            item['model_output']=request_model(item['prompt'])
        retry=5
        while 'model_output' in item and (len(item['model_output'])>7 or '+' in item['model_output']):
            print(item['model_output'])
            item['model_output']=request_model(item['prompt'])
            print('retry due to CoT!')
            retry-=1
            if retry==0:
                break
    if not os.path.exists('./data/result'):  
        os.makedirs('./data/result')  
    with open(f'./data/result/{args.model}_step{args.step}_{args.order}.json','w') as f:
        json.dump(test_data,f,indent=4,ensure_ascii=False)
    
    cnt=[]
    for item in test_data:
        try:
            if '=' in item['model_output']:
                if(int(item['gt'])==int(item['model_output'].strip('.').split('=')[1])):
                    cnt.append(str(item['var_as_sub']))
                else:
                    cnt.append(str(item['var_as_sub'])+'-')
            else:
                if(int(item['gt'])==int(item['model_output'].strip('.'))):
                    cnt.append(str(item['var_as_sub']))
                else:
                    cnt.append(str(item['var_as_sub'])+'-')
        except:
            # print(item['model_output'])
            continue
    
    print('===stat===')
    for i in range(0,args.step):
        print(f"step{i}:","{:.2f} ({}/{})".format(Counter(cnt)[str(i)]/(Counter(cnt)[str(i)]+Counter(cnt)[str(i)+'-']),Counter(cnt)[str(i)],Counter(cnt)[str(i)]+Counter(cnt)[str(i)+'-']))