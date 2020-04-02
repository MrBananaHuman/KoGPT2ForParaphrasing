from itertools import permutations

data = open('paraKQC_v1.txt', 'r', encoding='utf-8')
data2 = open('kor_pair_train.csv', 'r', encoding='utf-8')

output = open('para_sentence_pair.txt', 'w', encoding='utf-8')

lines = data.readlines()
data.close()

def write_combi_pair(sentences):
    combi_list = list(permutations(sentences, 2))
    for combi in combi_list:
        output.write(combi[0] + '\t' + combi[1])
        output.write('\n')

similar_sents = []
for line in lines:
    line = line.replace('\n', '')
    sent = line.split('\t')[2]
    similar_sents.append(sent)
    if len(similar_sents) == 10:
        write_combi_pair(similar_sents)
        similar_sents = []
        
lines = data2.readlines()
data2.close()

for line in lines:
    line = line.replace('\n', '') 
    if line.split(',')[5] == '0':
        output.write(line.split(',')[3] + '\t' + line.split(',')[4])
        output.write('\n')
        
output.close()
