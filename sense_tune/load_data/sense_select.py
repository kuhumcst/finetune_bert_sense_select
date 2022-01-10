from collections import namedtuple
from typing import List
import torch
import random


class Sense_Selection_Data(List):
    def __init__(self, data, tokenizer, max_seq_length=254, pad_token=0,
                 mask_zero_padding=True,
                 emb_model=None, linear=False, data_type='training'):
        self.max_seq_length = max_seq_length
        self.pad_token = pad_token
        self.emb_model = emb_model
        self.linear = linear
        self.mask_zero_padding = mask_zero_padding

        if data_type == 'reduce':
            super().__init__(self.load_reduction_data(data, tokenizer))
        else:
            super().__init__(self.load_data2(data, tokenizer))

    @staticmethod
    def truncate_pair_to_max_length(tokens_a, tokens_b, max_length):
        total_length = len(tokens_a) + len(tokens_b)
        while total_length > max_length:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_data(self, data, tokenizer):
        datapoints = []
        BertInput = namedtuple("BertInput", ["lemma", "row",
                                             "input_ids", "input_mask",
                                             "segment_ids", "label_id"])


        for row in data.itertuples():
            tokens_a = tokenizer.tokenize(row.sentence)

            sequences = [(sent2, 1 if i == row.target else 0) for i, sent2 in enumerate(row.examples)]

            # lab_1 = [str(i) for i, sent2 in enumerate(row.examples) if str(i) in row.target]

            # for lab in lab_1:
            pairs = []
            for ind, (sent2, label) in enumerate(sequences):
                # if ind != int(lab) and label == 1:
                #   continue
                tokens_b = tokenizer.tokenize(sent2)

                self.truncate_pair_to_max_length(tokens_a, tokens_b, self.max_seq_length - 3)

                # add first sentence
                tokens_sent_1 = tokens_a + ['[SEP]']
                sent_1_ids = [0] * len(tokens_sent_1)

                # add secound sentence
                tokens_sent_2 = tokens_b + ['[SEP]']
                sent_2_ids = [1] * (len(tokens_sent_2))

                all_tokens = ['[CLS]'] + tokens_sent_1 + tokens_sent_2
                ids = [1] + sent_1_ids + sent_2_ids

                input_ids = tokenizer.convert_tokens_to_ids(all_tokens)

                input_mask = [1 if self.mask_zero_padding else 0] * len(input_ids)

                # Zero pad to the max_seq_length.
                pad_length = self.max_seq_length - len(input_ids)

                input_ids = input_ids + ([self.pad_token] * pad_length)
                input_mask = input_mask + ([0 if self.mask_zero_padding else 1] * pad_length)
                segment_ids = ids + ([0] * pad_length)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                pairs.append(BertInput(lemma=row.lemma,
                                       row=row.Index,
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_id=label))

                if self.linear is True:
                    datapoints.append(pairs)
                    pairs = []

            if self.linear is False:
                datapoints.append(pairs)

        return datapoints

    def load_data2(self, data, tokenizer):
        datapoints = []
        BertInput = namedtuple("BertInput", ["lemma", "row",
                                             "input_ids", "input_mask",
                                             "segment_ids", "label_id"])
        random.seed(123)

        for row in data.itertuples():
            tokens_a = tokenizer.tokenize(row.sentence)

            target_index = [i for i, sent2 in enumerate(row.examples) if i == row.target][0]

            length = len(row.examples) - 1
            indexes = random.sample([i for i, sent2 in enumerate(row.examples) if i != target_index],
                                    length if length < 2 else 2)
            indexes.append(target_index)

            sequences = [(sent2, 1 if i == row.target else 0) for i, sent2 in enumerate(row.examples) if i in indexes]

            # lab_1 = [str(i) for i, sent2 in enumerate(row.examples) if str(i) in row.target]
            pairs = []
            # for lab in lab_1:
            for ind, (sent2, label) in enumerate(sequences):
                if ind not in indexes:
                    continue

                # if ind != int(lab) and label == 1:
                #   continue
                tokens_b = tokenizer.tokenize(sent2)

                self.truncate_pair_to_max_length(tokens_a, tokens_b, self.max_seq_length - 3)

                # add first sentence
                tokens_sent_1 = tokens_a + ['[SEP]']
                sent_1_ids = [0] * len(tokens_sent_1)

                # add secound sentence
                tokens_sent_2 = tokens_b + ['[SEP]']
                sent_2_ids = [1] * (len(tokens_sent_2))

                all_tokens = ['[CLS]'] + tokens_sent_1 + tokens_sent_2
                ids = [1] + sent_1_ids + sent_2_ids

                input_ids = tokenizer.convert_tokens_to_ids(all_tokens)

                input_mask = [1 if self.mask_zero_padding else 0] * len(input_ids)

                # Zero pad to the max_seq_length.
                pad_length = self.max_seq_length - len(input_ids)

                input_ids = input_ids + ([self.pad_token] * pad_length)
                input_mask = input_mask + ([0 if self.mask_zero_padding else 1] * pad_length)
                segment_ids = ids + ([0] * pad_length)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                pairs.append(BertInput(lemma=row.lemma,
                                       row=row.Index,
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       segment_ids=segment_ids,
                                       label_id=label))

                #if self.linear is True:
                datapoints.append(pairs)
                pairs = []

            #if self.linear is False:
                #datapoints.append(pairs)

        return datapoints

    def load_reduction_data(self, data, tokenizer):
        datapoints = []
        BertInput = namedtuple("BertInput", ["lemma", "row",
                                             "input_ids", "input_mask",
                                             "segment_ids", "label_id"])

        for row in data.itertuples():
            pairs = []
            tokens_a = tokenizer.tokenize(row.sentence_1)
            tokens_b = tokenizer.tokenize(row.sentence_2)

            self.truncate_pair_to_max_length(tokens_a, tokens_b, self.max_seq_length - 3)

            # add first sentence
            tokens_sent_1 = tokens_a + ['[SEP]']
            sent_1_ids = [0] * len(tokens_sent_1)

            # add secound sentence
            tokens_sent_2 = tokens_b + ['[SEP]']
            sent_2_ids = [1] * (len(tokens_sent_2))

            all_tokens = ['[CLS]'] + tokens_sent_1 + tokens_sent_2
            ids = [1] + sent_1_ids + sent_2_ids

            input_ids = tokenizer.convert_tokens_to_ids(all_tokens)
            input_mask = [1 if self.mask_zero_padding else 0] * len(input_ids)

            # Zero pad to the max_seq_length.
            pad_length = self.max_seq_length - len(input_ids)

            input_ids = input_ids + ([self.pad_token] * pad_length)
            input_mask = input_mask + ([0 if self.mask_zero_padding else 1] * pad_length)
            segment_ids = ids + ([0] * pad_length)

            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length

            pairs.append(BertInput(lemma=row.lemma,
                                   row=row.Index,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   segment_ids=segment_ids,
                                   label_id=row.label))

            datapoints.append(pairs)

        return datapoints


def collate_batch(batch):
    max_seq_length = len(batch[0][0].input_ids)

    collated = []
    for sub_batch in batch:
        batch_size = len(sub_batch)

        id_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        mask_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        segment_collated = torch.zeros([batch_size, max_seq_length], dtype=torch.long)
        label_collated = torch.zeros([batch_size], dtype=torch.long)

        for i, bert_input in enumerate(sub_batch):
            id_collated[i] = torch.tensor(bert_input.input_ids, dtype=torch.long)
            mask_collated[i] = torch.tensor(bert_input.input_mask, dtype=torch.long)
            segment_collated[i] = torch.tensor(bert_input.segment_ids, dtype=torch.long)
            label_collated[i] = torch.tensor(bert_input.label_id, dtype=torch.long)

        collated.append([bert_input.lemma,
                         bert_input.row,
                         id_collated, mask_collated, segment_collated, label_collated])

    return collated


class SentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
