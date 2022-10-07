from sense_tune.model.save_checkpoints import load_checkpoint
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
import torch


def get_token_repr_idx(input_ids: torch.tensor):
    """get the placement of [TGT] token in the bert input"""
    batch_size = input_ids.shape[0]
    placement = input_ids == 31748  # todo: make variable
    token_idxs = placement.nonzero().transpose(1, 0)
    return [token_idxs[1][token_idxs[0] == b] for b in range(batch_size)]


def get_repr_avg(output_hidden_states, token_idx, n_sent=1):
    """calculate the average representation for the tokens in range token_idx for the last four layers.
    if multiple tokens are in range token_idx, then the final representation is:
        average of the four last layers --> average of tokens --> final representation

    :param output_hidden_states: Bert output
    :param token_idx: (list) start and end index for target tokens
    :param n_sent: (int) 1 sentence or 2 sentence input
    """
    layers_hidden = output_hidden_states[-4:]  # get four last hidden layers
    layers_hidden = torch.mean(torch.stack(layers_hidden), axis=0)  # first average (hidden layers)

    batch_size = layers_hidden.shape[0]
    hidden_token = [layers_hidden[b, token_idx[b][0]:token_idx[b][1] - 1, :]
                    for b in range(batch_size)]  # get target tokens for each instance in batch

    hidden_token = torch.stack([torch.mean(hidden, dim=0)
                                if hidden.shape[0] > 1 else hidden.squeeze(0)
                                for hidden in hidden_token]).reshape(batch_size, -1, 768)  # second average (token)

    if hidden_token.shape[0] > n_sent * 2:  # if multiple [TGT] are present, then only use the first one
        hidden_token_1 = hidden_token[:2]
        hidden_token_2 = hidden_token[-2:]
        hidden_token = torch.concat((hidden_token_1, hidden_token_2), dim=1)

    # return hidden_token.reshape(batch_size, 2, 768)
    return hidden_token.reshape(batch_size, n_sent * 768)

class BertSense(BertPreTrainedModel):
    """
    BERT model for sense similarity / proximity estimation using CLS token
    Inherits from BERT

    Attributes
    ----------
    :attr num_labels: number of labels for training
    :attr config: bert config
    :attr dropout: dropout level
    :attr sigmoid: sigmoid activation function
    :attr relu: Leaky ReLU activation function
    :attr out: linear output layer
    :attr softmax: softmax function

    Methods
    -------
    forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.relu = torch.nn.LeakyReLU()
        self.out = torch.nn.Linear(self.config.hidden_size, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function for sense similarity / proximity estimation"""
        # import pdb; pdb.set_trace()
        bert_out = self.bert(input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )
        # returns the last hidden layer of the classification token further processed by a relu activation function
        # and a linear layer
        bert_out = self.dropout(bert_out[1])
        class_out = self.out(self.relu(bert_out))
        class_out = self.softmax(class_out)

        return class_out.squeeze()[0]


class BertSenseToken(BertPreTrainedModel):
    """
    BERT model for sense similarity / proximity estimation using average target token embedding
    Inherits from BERT

    Attributes
    ----------
    :attr num_labels: number of labels for training
    :attr config: bert config
    :attr dropout: dropout level
    :attr sigmoid: sigmoid activation function
    :attr cos: cosine similarity function
    :attr activation1: Leaky ReLU activation function for the first postprocessing layer
    :attr activation2: TanH actionvation function for the second postprocessing layer
    :attr reduce: first postprocessing linear layer that reduces nodes from hidden_size to 192
    :attr combine: second postprocessing linear layer that combines the two sentence embeddings
    :attr out: linear output layer

    Methods
    -------
    forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score using target token embedding

    forward_cos(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None)
        :returns: Bert sense similarity / proximity score using cosine similarity
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = torch.nn.CosineSimilarity(dim=1)

        self.activation1 = torch.nn.LeakyReLU()
        self.activation2 = torch.nn.Tanh()
        self.reduce = torch.nn.Linear(config.hidden_size, 192)
        self.combine = torch.nn.Linear(192 * 2, 192)
        self.out = torch.nn.Linear(192, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function for sense similarity / proximity estimation"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_ids = get_token_repr_idx(input_ids)  # get target token placement
        batch_size = input_ids.shape[0]
        bert_out = self.bert(input_ids[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             attention_mask=attention_mask[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             token_type_ids=token_type_ids[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )

        hidden_states = bert_out.hidden_states
        # retrieve the target token (placement known from [TGT])
        new_output = get_repr_avg(hidden_states, token_ids, n_sent=2)

        bert_out = self.dropout(new_output)

        # returns the last hidden layer of the classification token further processed by a Linear layer
        # and a leaky ReLU activation function
        linear = self.reduce(self.activation1(bert_out))

        # combines the two token representations through a linear layer + a Tanh activation function
        linear = self.combine(self.activation2(linear))
        # class_out = model.out(linear)
        class_out = self.softmax(self.out(linear))
        class_out = self.softmax(class_out)

        return class_out.squeeze()[0]

    def forward_cos(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                    inputs_embeds=None, output_attentions=None, return_dict=None):
        """updated forward function using cosine similarity instead of trained postprocessing
        #####THIS HAS NOT BEEN TESTED#####
        """

        # import pdb; pdb.set_trace()
        token_ids = get_token_repr_idx(input_ids)  # get target token placement
        batch_size = input_ids.shape[0]
        bert_out = self.bert(input_ids[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             attention_mask=attention_mask[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             token_type_ids=token_type_ids[input_ids != 31748].view(batch_size, -1),  # remove [TGT] token
                             position_ids=position_ids,
                             head_mask=head_mask,
                             inputs_embeds=inputs_embeds,
                             output_attentions=output_attentions,
                             output_hidden_states=True,
                             return_dict=return_dict
                             )

        bert_out = self.dropout(bert_out)
        hidden_states = bert_out.hidden_states
        # retrieve the target token (placement known from [TGT])
        new_output = get_repr_avg(hidden_states, token_ids)
        # reduce dimensionality
        new_output = self.reduce(self.activation1(new_output))
        class_out = self.cos(new_output[:, :192], new_output[:, 192:])

        return class_out


def get_model_and_tokenizer(model_name, model_type, device, checkpoint=False):
    # if args.local_rank not in [-1, 0]:
    # Make sure only the first process in distributed training will download model & vocab
    # torch.distributed.barrier()

    config = BertConfig.from_pretrained(model_name, num_labels=2)

    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    if model_type == 'bert_token':
        model = BertSenseToken.from_pretrained(model_name, config=config)
    elif model_type == 'bert_token_cos':
        model = BertSenseToken.from_pretrained(model_name, config=config)
        model.forward = model.forward_cos
    else:
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
