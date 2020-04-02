import gluonnlp as nlp
from gluonnlp.data import SentencepieceTokenizer
from tokenizers import ByteLevelBPETokenizer

from transformers.tokenization_utils import PreTrainedTokenizer, PreTrainedTokenizerFast

import json

class MyTokenizer():

    def __init__(self, vocab_file_path):
        self.tokenizer = SentencepieceTokenizer(vocab_file_path)
        self.unknown_token = self.tokenizer.tokens.index("<unk>")
        self._pad_token = "<pad>"
        self.pad_token_id = self.tokenizer.tokens.index("<pad>")
        self.max_len = 1024
        self.max_len_single_sentence = 1024
        self.init_kwargs = {}
        self.added_tokens_encoder = {}
        self.unique_added_tokens_encoder = set()
        self.added_tokens_decoder = {}
        self.unexpected_sep_token = ['<pad>', '<unk>']
        self.vocab_b_obj = vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(vocab_file_path,
                                                         mask_token=None,
                                                         sep_token=None,
                                                         cls_token=None,
                                                         unknown_token='<unk>',
                                                         padding_token='<pad>',
                                                         bos_token='<s>',
                                                         eos_token='</s>')
        self.vocab_handle = self.vocab_b_obj.to_json()
        self.encoder = json.loads(self.vocab_handle)

    
    def tokenize(self, text):
        if text in self.unexpected_sep_token:
            return text
        return self.tokenizer(text)
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        if isinstance(tokens, str):
            if tokens in self.encoder['token_to_idx']:
                return self.encoder['token_to_idx'][tokens]
            else:
                return self.unknown_token
        for token in tokens:
            if token in self.encoder['token_to_idx']:
                ids.append(self.encoder['token_to_idx'][token])
            else:
                ids.append(self.unknown_token)
        return ids
    
    def convert_ids_to_tokens(self, ids):
        sentence = ''
        for id_ in ids:
            sentence += self.encoder['idx_to_token'][id_]
        sentence = sentence.replace('▁', ' ')
        return sentence.strip()
            
    
    def build_inputs_with_special_tokens(self, ids):
        return ids

    def get_vocab_size(self):
        return len(self.tokenizer.tokens)

    def add_special_tokens(self, new_tokens):
        cnt = 0
        for token_list in new_tokens:
            tokens = new_tokens[token_list]
            if isinstance(tokens, str):
                if tokens not in self.encoder['token_to_idx']:
                    last_num = len(self.encoder['token_to_idx'].keys())
                    self.encoder['token_to_idx'][tokens] = last_num
                    self.encoder['idx_to_token'].append(tokens)

                    print(tokens, ' token is added in vocab!')
                    print('>>>', self.encoder['token_to_idx'][tokens])
                    cnt += 1
                else:
                    print(tokens, ' token is already in vocab')
                    print('>>>', self.encoder['token_to_idx'][tokens])
            else:
                for token in tokens:
                    if token not in self.encoder['token_to_idx']:
                        last_num = len(self.encoder['token_to_idx'].keys())
                        self.encoder['token_to_idx'][token] = last_num
                        self.encoder['idx_to_token'].append(token)

                        print(token, ' token is added in vocab!')
                        print('>>>', self.encoder['token_to_idx'][token])
                        cnt += 1
                    else:
                        print(token, ' token is already in vocab')
                        print('>>>', self.encoder['token_to_idx'][token])
        return cnt
    # def save_pretrained(self, save_directory)
        # self.tokenizer.

if __name__ == '__main__':
    vocab_file_path = 'model/kogpt2_news_wiki_ko_cased_818bfa919d.spiece'
    tokenizer = MyTokenizer(vocab_file_path)
    sentence = "이순신은 조선 중기의 무신이다."
    tokens = tokenizer.tokenize(sentence)
    print(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)
    ids2 = tokenizer.build_inputs_with_special_tokens(ids)
    print(ids2)
    print(tokenizer.convert_ids_to_tokens(ids))
    
