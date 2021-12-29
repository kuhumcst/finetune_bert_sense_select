import torch
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer
from sense_tune.model.save_checkpoints import load_checkpoint

BERT_MODEL = 'Maltehb/danish-bert-botxo'


class BertSense(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
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
