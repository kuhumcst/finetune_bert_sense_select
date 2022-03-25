import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from sense_tune.model.save_checkpoints import load_checkpoint

BERT_MODEL = 'Maltehb/danish-bert-botxo'




class BertSense(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Tanh() #torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()


def forward(model, batch, device):
    batch = tuple(tensor.to(device) for tensor in batch)
    # import pdb; pdb.set_trace()

    bert_out = model.bert(input_ids=batch[0], attention_mask=batch[1], token_type_ids=batch[2])

    # returns the last hidden layer of the classification token further processed by a Linear layer
    # and a Tanh activation function
    bert_out = model.dropout(bert_out[1])
    # linear = model.relu(model.linear(bert_out))
    # class_out = model.out(linear)
    class_out = model.out(bert_out)

    return class_out.squeeze(-1)


class BertSenseToken(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(config.hidden_size*2, 100)
        self.out = torch.nn.Linear(100, 1)
        self.init_weights()


def forward2(model, batch, device):
    batch = tuple(tensor.to(device) for tensor in batch)
    # import pdb; pdb.set_trace()

    bert_out = model.bert(input_ids=batch[0],
                          attention_mask=batch[1],
                          token_type_ids=batch[2],
                          output_hidden_states=True
                          )

    def get_token_repr_idx(input_ids: torch.tensor):
        batch_size = input_ids.shape[0]
        placement = input_ids == 31748
        token_idxs = placement.nonzero().transpose(1,0)
        return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]

    def get_repr_avg(output_hidden_states, token_idx, layers):
        layers_hidden = output_hidden_states[layers]

        if isinstance(layers_hidden, tuple):
            layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)

        batch_size = layers_hidden.shape[0]
        hidden_token = [layers_hidden[b, token_idx[b][i]+1:token_idx[b][j], :]
                        for b in range(batch_size) for i, j in [(0, 1), (-2, -1)]
                        ]
        hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                    if hidden.shape[0] > 1 else hidden.squeeze(0)
                                    for hidden in hidden_token])

        if hidden_token.shape[0] % 2 != 0:
            hidden_token = hidden_token[::2]

        return hidden_token.reshape(batch_size, 768*2)

    hidden_states = bert_out.hidden_states
    token_ids = get_token_repr_idx(batch[0])
    new_output = get_repr_avg(hidden_states, token_ids, layers=-1)

    # returns the last hidden layer of the classification token further processed by a Linear layer
    # and a Tanh activation function
    #bert_out = model.dropout(bert_out[1])
    bert_out = model.dropout(new_output)
    linear = model.relu(model.linear(bert_out))
    # class_out = model.out(linear)
    class_out = model.out(bert_out)

    return class_out.squeeze(-1)


def get_model_and_tokenizer(model_name, device, checkpoint=False):
    # if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    # torch.distributed.barrier()

    config = BertConfig.from_pretrained(model_name, num_labels=2)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    model = BertSense.from_pretrained(model_name, config=config)

    # add new special token
    if '[TGT]' not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        assert '[TGT]' in tokenizer.additional_special_tokens
        model.resize_token_embeddings(len(tokenizer))

    if checkpoint and checkpoint != 'None':
        model, loss = load_checkpoint(checkpoint, model, device)

    model.to(device)

    return model, tokenizer
