import sys
import torch

from torch.utils.data import DataLoader
import sense_tune.utils as utils
from sense_tune.load_data.sense_select import Sense_Selection_Data, SentDataset, collate_batch
from sense_tune.model.bert import get_model_and_tokenizer
from sense_tune.model.save_checkpoints import save_checkpoint, save_metrics
from sense_tune.model.train import train, evaluate


def main(num_epochs, training, testing):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set fixed random number seed
    # torch.manual_seed(42)

    for learning_rate in [0.00002]:

        model_name = 'Maltehb/danish-bert-botxo'
        model, tokenizer = get_model_and_tokenizer(model_name, device)

        print('Loading data...')
        training = SentDataset(Sense_Selection_Data(training, tokenizer))

        train_loader = DataLoader(training,
                                 batch_size=1,
                                 shuffle=True,
                                 collate_fn=collate_batch)

        print('Loading test')
        testing = SentDataset(Sense_Selection_Data(testing, tokenizer))
        test_loader = DataLoader(testing,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate_batch)


        for epoch in range(num_epochs):
            print(f'Epoch number: {epoch+1} of {num_epochs}')
            print('Training model...\n')
            n_steps, loss = train(model,
                                  train_loader,
                                  device,
                                  learning_rate=learning_rate,
                                  num_epochs=1)

            print(f'Total number of training steps: {n_steps}')
            print(f'Training loss: {loss}')

            print('\nEvaluating model on training...')
            train_loss, train_accuracy = evaluate(model, train_loader, device)
            # Print accuracy
            print(f'Train-evaluation loss: {train_loss}')
            print(f'Train accuracy: {100 * train_accuracy}')

            print('\nEvaluating model on test data...')
            test_loss, test_accuracy = evaluate(model, test_loader, device)

            # Print accuracy
            print(f'Test loss: {test_loss}')
            print(f'Test accuracy: {100 * test_accuracy}')
            print('--------------------------------')

        save_checkpoint('sense_tune/model/checkpoints/model_bert.pt', model, train_loss)
        save_metrics('sense_tune/model/checkpoints/metrics_bert.pt',
                     train_loss, loss, n_steps)



if __name__ == "__main__":
    main(num_epochs=int(sys.argv[2]),
         training=utils.load_datapoints_from_path(sys.argv[3], sys.argv[1]),
         testing=utils.load_datapoints_from_path(sys.argv[4], sys.argv[1])  # sys.argv[1])
         )
