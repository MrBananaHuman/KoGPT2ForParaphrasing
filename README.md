# KoGPT2ForPara

GPT2 model: https://github.com/SKT-AI/KoGPT2

Paraphrasing data: https://github.com/warnikchow/paraKQC, https://github.com/songys/Question_pair

GPT2 finetuning code: https://github.com/huggingface/transfer-learning-conv-ai

Train script:
python3 train.py --dataset_path=data/train_data --init_model=model/pytorch_kogpt2_676e9bcfa7.params --n_epochs=1
