import random
import re
from tqdm import tqdm  
import argparse
import json
import os

def is_integer_in_range(number):
    if isinstance(number, int) and 0 <= number <= 22:
        return True
    return False

def gen(step):
    while True:
        try:
            code=[]
            for i in range(step):
                var=f'num{i}_'
                op=random.choice('+-')
                if i==0:
                    num1=random.randint(0,22)
                    num2=random.randint(0,22)
                else:
                    if random.randint(0,1):
                        num1=random.choice(list(range(23)))
                        num2=f'num{i-1}_'
                    else:
                        num2=random.choice(list(range(23)))
                        num1=f'num{i-1}_'
                code.append(var+' = '+str(num1)+' '+op+' '+str(num2))
                exe_code=''
                for line in code:
                    exe_code+=line+'\n'
                exec(exe_code)
                assert(is_integer_in_range(eval(var)))
            return code,eval(var)
        except Exception as e:
            # print(e)
            pass
            
def get_num(data):
    pattern = r'\d+ - [a-zA-Z]'
    lines = data.strip().split('\n')
    count = sum(1 for line in lines if re.search(pattern, line))
    return count

def replace(input_text,num_list):
    for i in range(26):
        input_text = input_text.replace(f'num{i}_', num_list[i])
    return input_text

def main(args):
    step=args.step
    count_dict = {}  
    for item in range(step):
        count_dict[item]=0
    total_target = len(count_dict) * 100 

    template = []  
    data_dup = []  
    progress_bar = tqdm(total=total_target, desc="Generating Data")  
    while any(count < 100 for count in count_dict.values()):  
        gen_code, val = gen(step)  
        if '\n'.join(gen_code) in data_dup:  
            continue  
        for item in gen_code[-1:]:  
            eqs = '\n'.join(gen_code)  
            count_value = get_num(eqs)  
            if count_value in count_dict and count_dict[count_value] < 100:  
                template.append({"premise": eqs, 'var_as_sub': count_value, 'gt': str(val % 23)})  
                count_dict[count_value] += 1  
                data_dup.append('\n'.join(gen_code))  
                progress_bar.update(1)  
    progress_bar.close()

    test_dataset=[]
    for item in template:
        alpha_list=random.sample(list('abcdefghijklmnopqrstuvwxyz'),26)
        test_dataset.append({'premise': replace(item['premise'], alpha_list), 'target':alpha_list[step-1], 'gt': item['gt'], 'var_as_sub': item['var_as_sub']})
    
    if not os.path.exists('./data'):  
        os.makedirs('./data')  
    with open(f'./data/testdata_step{str(step)}.json','w') as f:
        json.dump(test_dataset,f,indent=4,ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=3)
    args = parser.parse_args()

    main(args)