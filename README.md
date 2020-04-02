# KoGPT2ForParaphrasing

Korean paraphrasing test using KoGPT2 model

GPT2 model: https://github.com/SKT-AI/KoGPT2

Paraphrasing data: https://github.com/warnikchow/paraKQC, https://github.com/songys/Question_pair

preprocessing:

- make_sentence_pair.py

- make_train_data.py

Train script:

- python3 train.py --dataset_path=data/train_data --init_model=model/pytorch_kogpt2_676e9bcfa7.params --n_epochs=1


Test script:

- python3 interact.py --model=gpt2 --model_checkpoint=sample_model/pytorch_kogpt2_676e9bcfa7.params/pytorch_model.bin

sample model (ephoch number: 1)

- https://drive.google.com/open?id=1hyoWGQwKu4_nWdRWV2rLJPySC8pSRQWU

samples:


- 오늘도 좋은 하루 보내세요.

오늘 하루만 지나면 좋겠어.

오늘도 좋은 하루 보내세요.

오늘도 좋은 하루 보내세요.

오늘 하루종일 수고 많으셨어요.

오늘도 좋은 하루 만드세요.

오늘 하루종일 좋은 일만 있기를.

좋은 아침이에요. 오늘도 좋은 하루 보내세요.

좋은 아침이에요. 오늘도 좋은 하루 되세요.

오늘 하루종일 즐기자



- 오늘은 저녁 때 뭐 먹을거야?

오늘 저녁에 뭐 먹기로 했어?</s>

저녁 언제 먹기로 하셨나요?</s>

저녁 약속을 몇 시로 정하셨나요?</s>

저녁에는 뭐 먹지?</s>

오늘 저녁 약속 몇 신지 알려줘요</s>

저녁 먹고 나서 뭐 먹을까?</s>

오늘 저녁 언제 먹기로 했어?</s>

오늘 저녁에는 뭐 먹을까?</s>

저녁 약속은 몇 시에 가능하니?</s>

저녁밥 먹고 싶은데 가능한 시간이 언제야?</s>



- 맥주 한 잔 마실까?

맥주 한 잔 하고싶은데</s>

맥주 한 잔 어때?</s>

맥주 한잔 할까?</s>

맥주 한잔 할래?</s>

맥주 한잔하러 갈까?</s>

맥주 한잔 어때?</s>



- 집에서 라면 먹고 갈래?

집에서 라면 끓여서 먹으려면 어떻게 해야돼?</s>

라면 먹고 싶어 집에 가고 싶어</s>

집에서 라면 먹으려면 뭐라고 해야하니?</s>

라면 집에서 끓여먹나,</s>

집에서 라면 끓여서 먹기로 해놓고.</s>

집에서 라면 먹으려는데 어떻게 할까?</s>

라면 먹고 싶다</s>

집에서 라면 끓여먹을까?</s>

집에서 라면 먹고 싶어</s>

