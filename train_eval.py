import sys

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import sense_tune.utils as utils
from sense_tune.load_data.sense_select import Sense_Selection_Data, SentDataset, collate_batch
from sense_tune.model.bert import get_model_and_tokenizer
from sense_tune.model.reduce import get_BERT_score
from sense_tune.model.save_checkpoints import save_checkpoint, save_metrics
from sense_tune.model.train import train, evaluate


def k_fold_results(results: dict):
    # prints results from dict
    total = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        total += value
    print(f'Average: {total / len(results.items())} %')


def main(k_folds, num_epochs, training, testing=None,
         learning_rate=0.00002, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set fixed random number seed
    # torch.manual_seed(42)

    model_name = 'Maltehb/danish-bert-botxo'
    model, tokenizer = get_model_and_tokenizer(model_name, device)

    # Normal train + test evaluation
    if k_folds < 1:
        print('Loading data...')
        training = SentDataset(Sense_Selection_Data(training, tokenizer))

        data_loader = DataLoader(training,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=collate_batch)

        print('Training model...\n')
        n_steps, loss = train(model,
                              data_loader,
                              device,
                              learning_rate=learning_rate,
                              num_epochs=num_epochs)

        print(f'Total number of training steps: {n_steps}')
        print(f'Training loss: {loss}')

        print('\nEvaluating model on training...')
        train_loss, train_accuracy = evaluate(model, data_loader, device)
        # Print accuracy
        print(f'Train-evaluation loss: {train_loss}')
        print(f'Train accuracy: {100 * train_accuracy}')

        save_checkpoint('sense_tune/model/checkpoints/model_bert.pt', model, train_loss)
        save_metrics('sense_tune/model/checkpoints/metrics_bert.pt',
                     train_loss, loss, n_steps)

        print('loading test')
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

    # k-fold cross validation
    else:

        dataset = SentDataset(Sense_Selection_Data(training, tokenizer))

        # For fold results
        train_results = {}
        test_results = {}

        train_loss_list = []
        valid_loss_list = []
        global_steps_list = []
        kfold = KFold(n_splits=k_folds, shuffle=True)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            torch.cuda.empty_cache()
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=train_subsampler,
                                      collate_fn=collate_batch)

            test_loader = DataLoader(dataset,
                                     batch_size=1,
                                     sampler=test_subsampler,
                                     collate_fn=collate_batch)
            if fold > 0:
                model, tokenizer1 = get_model_and_tokenizer(model_name, device)

            print('Training model...\n')
            n_steps, loss = train(model,
                                  train_loader,
                                  device,
                                  learning_rate=learning_rate,
                                  num_epochs=num_epochs)

            print(f'Total number of training steps for fold {fold}: {n_steps}')
            print(f'Training loss for fold {fold}: {loss}')
            train_loss_list.append(loss)
            global_steps_list.append(n_steps)

            print('\nEvaluating model on training...')
            train_loss, train_accuracy = 0, 0  # evaluate(model, train_loader, device)
            train_results[fold] = 100.0 * train_accuracy
            valid_loss_list.append(train_loss)

            # Print accuracy
            print(f'Train-evaluation loss for fold {fold}: {train_loss}')
            print(f'Train accuracy for fold {fold}: {100 * train_accuracy}')

            print('\nEvaluating model on test data...')
            test_loss, test_accuracy = evaluate(model, test_loader, device)

            # Print accuracy
            print(f'Test loss for fold {fold}: {test_loss}')
            print(f'Test accuracy for fold {fold}: {100 * test_accuracy}')
            test_results[fold] = 100.0 * test_accuracy

            save_checkpoint('sense_tune/model/checkpoints' + '/model_' + str(fold) + '.pt', model, train_loss)
            save_metrics('sense_tune/model/checkpoints/metrics_' + str(fold) + '.pt',
                         train_loss_list, valid_loss_list, global_steps_list)

            print('--------------------------------')

        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS (TESTING)')
        print('--------------------------------')
        k_fold_results(test_results)
        print('--------------------------------')

        print(f'K-fold cross validation results for {k_folds} folds (training)')
        print('--------------------------------')
        k_fold_results(train_results)


if __name__ == "__main__":
    if len(sys.argv) <= 7:
        test = None
        reduction = sys.argv[6]
    else:
        test = sys.argv[6]
        reduction = sys.argv[7]

    main(num_epochs=int(sys.argv[3]),
         k_folds=int(sys.argv[4]),
         training=utils.load_datapoints_from_path(sys.argv[5], sys.argv[1]),
         testing=utils.load_datapoints_from_path(test, sys.argv[1])  # sys.argv[1])
         )

    run_id = str(sys.argv[2]),
    reduction_data = pd.read_csv(reduction, sep='\t', index_col=0)
    reduction_data_score = get_BERT_score(reduction_data)
    reduction_data_score.to_csv(f'/content/drive/MyDrive/SPECIALE/data/reduction_score_{run_id}.tsv', sep='\t')
