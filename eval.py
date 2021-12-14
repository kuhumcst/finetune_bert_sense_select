import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
import sense_tune.utils as utils
from sense_tune.load_data.sense_select import Sense_Selection_Data, SentDataset, collate_batch
from sense_tune.model.bert import get_model_and_tokenizer
from sense_tune.model.train import evaluate

def load_datapoints_from_path(path, dataset):
    # loads and precrocess the data
    if not path:
        return None

    if dataset == 'sense_select':

        datapoints = pd.read_csv(path, sep='\t', index_col=0)
        datapoints['examples'] = datapoints['examples'].apply(utils.process)
        datapoints['senses'] = datapoints['senses'].apply(utils.process)
        # datapoints['target'] = datapoints['target'].apply(process)
        return datapoints

    elif dataset == 'wic':
        datapoints = pd.read_csv(path, sep='\t')
        return datapoints

    else:
        return None


def main(testing):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set fixed random number seed
    # torch.manual_seed(42)

    model_name = 'Maltehb/danish-bert-botxo'
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    # Just test evaluation

    print('Loading data...')
    testing = SentDataset(Sense_Selection_Data(testing, tokenizer))

    data_loader = DataLoader(testing,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_batch)

    print('\nEvaluating model on test data...')
    test_loss, test_accuracy = evaluate(model, data_loader, device)

    # Print accuracy
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {100 * test_accuracy}')
    print('--------------------------------')


if __name__ == "__main__":
    main(testing=load_datapoints_from_path(sys.argv[1], 'sense_select'))
