from transformers import BatchEncoding, PreTrainedTokenizer, DataCollatorForWholeWordMask, BertTokenizer, BertTokenizerFast
from transformers.data.data_collator import _torch_collate_batch
import warnings
from torch.utils.data import Dataset
from datasets import load_dataset
import random
import torch
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


class MLMDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, num_processes=16):
        self.tokenizer = tokenizer
        self.max_length = self.tokenizer.model_max_length - self.tokenizer.num_special_tokens_to_add(pair=False)
        self.corpus = load_dataset(
            'HuggingFaceFW/fineweb',  # hard code for now
            'sample-100BT',
            split='train',
            num_proc=num_processes,
        )

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item) -> BatchEncoding:
        text = self.corpus[item]['text']

        # if the text is too long, truncate it randomly
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_length:
            trunc = len(tokens) - self.max_length
            trunc_left = random.randint(0, trunc)
            trunc_right = trunc - trunc_left

            truncated = tokens[trunc_left:]
            if trunc_right > 0:
                truncated = truncated[:-trunc_right]
            text = self.tokenizer.convert_tokens_to_string(truncated)

        tokenized_text = self.tokenizer(text,
                                        return_special_tokens_mask=False,
                                        return_token_type_ids=False,
                                        truncation=True)
        return tokenized_text

@dataclass
class DataCollatorForWholeWordMaskWithAttentionMask(DataCollatorForWholeWordMask):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        output = super().torch_call(examples)
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        output['attention_mask'] = batch['attention_mask']
        return output



@dataclass
class MaeDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    encoder_mlm_probability: float = 0.3
    decoder_mlm_probability: float = 0.5

    def __post_init__(self):
        super(MaeDataCollatorForWholeWordMask, self).__post_init__()

        from transformers import BertTokenizer, BertTokenizerFast
        from transformers import RobertaTokenizer, RobertaTokenizerFast
        if isinstance(self.tokenizer, (BertTokenizer, BertTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_bert
        elif isinstance(self.tokenizer, (RobertaTokenizer, RobertaTokenizerFast)):
            self.whole_word_cand_indexes = self._whole_word_cand_indexes_roberta
        else:
            raise NotImplementedError(f'{type(self.tokenizer)} collator not supported yet')

        self.specials = self.tokenizer.all_special_tokens

    def _whole_word_cand_indexes_bert(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_cand_indexes_roberta(self, input_tokens: List[str]):
        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in self.specials:
                raise ValueError('We expect only raw input for roberta for current implementation')

            if i == 0:
                cand_indexes.append([0])
            elif not token.startswith('\u0120'):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])
        return cand_indexes

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = self._whole_word_cand_indexes_bert(input_tokens)

        random.shuffle(cand_indexes)
        encoder_num_to_predict = min(max_predictions,
                                     max(1, int(round(len(input_tokens) * self.encoder_mlm_probability))))
        decoder_num_to_predict = min(max_predictions,
                                     max(1, int(round(len(input_tokens) * self.decoder_mlm_probability))))

        masked_lms = []
        encoder_masked_lms = []
        decoder_masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= max(encoder_num_to_predict, decoder_num_to_predict):
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > max(encoder_num_to_predict, decoder_num_to_predict):
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue

            encoder_add = True if len(encoder_masked_lms) + len(index_set) <= encoder_num_to_predict else False
            decoder_add = True if len(decoder_masked_lms) + len(index_set) <= decoder_num_to_predict else False

            for index in index_set:
                covered_indexes.add(index)
                if encoder_add and decoder_add:
                    encoder_masked_lms.append(index)
                if decoder_add:
                    decoder_masked_lms.append(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        encoder_mask_labels = [1 if i in encoder_masked_lms else 0 for i in range(len(input_tokens))]
        decoder_mask_labels = [1 if i in decoder_masked_lms else 0 for i in range(len(input_tokens))]
        return encoder_mask_labels, decoder_mask_labels

    def __call__(self, examples, return_tensors=None):
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            return_attention_mask=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        input_ids = [e["input_ids"] for e in examples]

        encoder_mlm_masks = []
        decoder_mlm_masks = []

        for e in input_ids:
            tokens = []
            for tid in e:
                tokens.append(self.tokenizer._convert_id_to_token(tid))

            encoder_mlm_mask, decoder_mlm_mask = self._whole_word_mask(tokens, self.tokenizer.model_max_length)
            encoder_mlm_masks.append(encoder_mlm_mask)
            decoder_mlm_masks.append(decoder_mlm_mask)

        encoder_mlm_masks = _torch_collate_batch(encoder_mlm_masks, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        decoder_mlm_masks = _torch_collate_batch(decoder_mlm_masks, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)


        encoder_inputs, encoder_labels = self.torch_mask_tokens(
            batch['input_ids'].clone(),
            encoder_mlm_masks.clone()
        )

        decoder_inputs, decoder_labels = self.torch_mask_tokens(
            batch['input_ids'].clone(),
            decoder_mlm_masks.clone()
        )

        output = {
            "input_ids": encoder_inputs,
            "labels": encoder_labels,
            "decoder_input_ids": decoder_inputs,
            "decoder_labels": decoder_labels,
            "attention_mask": batch['attention_mask'],
        }

        return output