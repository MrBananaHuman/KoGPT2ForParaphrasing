# KoGPT2ForPara

GPT2 model: https://github.com/SKT-AI/KoGPT2

Paraphrasing data: https://github.com/warnikchow/paraKQC, https://github.com/songys/Question_pair

Train script:
python3 train.py --dataset_path=data/train_data --init_model=model/pytorch_kogpt2_676e9bcfa7.params --n_epochs=1


Test script:
python3 interact.py --dataset_path=data/train_sample2.txt --model=gpt2 --model_checkpoint=runs/Mar24_16-05-08_banana_model/pytorch_kogpt2_676e9bcfa7.params/pytorch_model.bin --device=cpu


