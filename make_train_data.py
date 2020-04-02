import json
from collections import OrderedDict
import random
 
original_data = open('para_sentence_pair.txt', 'r', encoding='utf-8')
data = open('train_data', 'w', encoding='utf-8')

lines = original_data.readlines()

original_data.close()


total_data = OrderedDict()
train_list = []
valid_list = []


cnt = 0
total_len = len(lines)

random.shuffle(lines)

def random_sampling(num_samples, except_sent):
    samples = []
    while(len(samples) < num_samples):
        random_idx = random.randint(0, len(lines)-1)
        random_line = lines[random_idx].replace('\n', '')
        sample = random_line.split('\t')[1]
        if sample != except_sent:
            samples.append(sample)
    return samples
    

for i, line in enumerate(lines):
    utterances = OrderedDict()
    train = OrderedDict()

    line = line.replace('\n', '')
    sent1 = line.split('\t')[0]
    sent2 = line.split('\t')[1]

    candidates = random_sampling(15, sent2)
    candidates.append(sent2)

    utterances['candidates'] = candidates
    utterances['history'] = [sent1]

    train['utterances'] = [utterances]

    if i < len(lines) * 0.95:
        train_list.append(train)
    else:
        valid_list.append(train)

total_data['train'] = train_list

total_data['valid'] = valid_list

data.write(json.dumps(total_data, ensure_ascii=False, indent="\t") )

data.close()
    
    
    
    
    

















