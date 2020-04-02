# KoGPT2ForPara

GPT2 model: https://github.com/SKT-AI/KoGPT2

Paraphrasing data: https://github.com/warnikchow/paraKQC, https://github.com/songys/Question_pair

Train script:
python3 train.py --dataset_path=data/train_data --init_model=model/pytorch_kogpt2_676e9bcfa7.params --n_epochs=1


Test script:
python3 interact.py --model=gpt2 --model_checkpoint=sample_model/pytorch_kogpt2_676e9bcfa7.params/pytorch_model.bin

sample model (ephoch number: 3)

samples:


