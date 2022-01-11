import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from sense_tune.model.bert import BertSense, forward


def train(model, train_dataloader, device, learning_rate=1e-4,
          num_epochs=5, max_steps=0, max_grad_norm=1.0):
    """ Fine-tune the model """
    if max_steps > 0:
        max_steps
        num_epochs = max_steps // len(train_dataloader) + 1

    writer = SummaryWriter(f'/content/drive/MyDrive/SPECIALE/data/')

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn.BCEWithLogitsLoss()
    bin_loss_function = torch.nn.MSELoss()
    global_step = 0
    tr_loss, tr2_loss = 0.0, 0.0
    correct = 0

    model.zero_grad()
    model.train()

    for epoch in range(num_epochs):
        # set_seed(args)  # Added here for reproducibility
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        with tqdm(train_dataloader, unit="batch", desc="Iteration") as epoch_iterator:
            # import pdb; pdb.set_trace()
            for step, batches in enumerate(epoch_iterator):
                epoch_iterator.set_description(f"Epoch {epoch}")

                # run model
                batch_loss = 0
                logits_list = []
                labels = batches[0][5]

                if len(batches) < 1:
                    continue

                for batch in batches:
                    logits = forward(model, batch[2:], device)

                    targets = torch.max(batch[5].to(device), -1).indices.to(device).detach()
                    batch_loss += loss_function(logits.unsqueeze(dim=0), targets.unsqueeze(dim=-1))

                    logits = model.sigmoid(logits)
                    batch_loss += bin_loss_function(logits, batch[5].to(torch.float).to(device))
                    predictions = torch.tensor([1 if pred >= 0.5 else 0 for pred in logits])

                logits_list.append(logits)

                correct += ((predictions == labels).sum().item() / len(labels)) / (global_step+1)
                accuracy = correct

                loss = batch_loss / len(batches)
                # loss2 = bin_loss / len(batches)

                loss.backward()

                # loss2.backward()

                #epoch_iterator.set_postfix(loss=loss.item(), accuracy=100. * accuracy)

                tr_loss += loss.item()
                # tr2_loss += loss2.item()
                writer.add_scalar('Loss/train', tr_loss, global_step)
                writer.add_scalar('Accuracy/train', accuracy, global_step)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                model.zero_grad()
                global_step += 1

                if 0 < max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < max_steps < global_step:
                break

    writer.close()
    return global_step, tr_loss / global_step


def evaluate(model, eval_dataloader, device):
    eval_loss = 0.0
    nb_eval_steps = 0
    accuracy = 0
    accuracy2 = 0

    loss_function = torch.nn.CrossEntropyLoss()
    bin_loss_function = torch.nn.MSELoss()
    all_labels = []
    predictions = []
    predictions2 = []

    iterator = tqdm(eval_dataloader, desc="Iteration")
    for batches in iterator:
        if len(batches) < 1:
            continue

        model.eval()
        with torch.no_grad():

            # run model
            batch_loss = 0
            logits_list = []

            for batch in batches:
                logits = forward(model, batch[2:], device)
                targets = torch.max(batch[5].to(device), -1).indices.to(device).detach()
                #batch_loss += loss_function(logits, targets)#, batch[3].to(device).detach())
                batch_loss += loss_function(logits.unsqueeze(dim=0), targets.unsqueeze(dim=-1))

                logits = model.sigmoid(logits)
                batch_loss += bin_loss_function(logits, batch[5].to(torch.float).to(device))

            logits_list.append(logits)

            prediction = torch.tensor([1 if pred >= 0.5 else 0 for pred in logits])
            labels = batches[0][5] if isinstance(model, BertSense) else [b[1] for b in batches]

            correct = (prediction == labels).sum().item()
            accuracy += correct / len(labels)

            loss = batch_loss / len(batches)

            eval_loss += loss
            prediction2 = logits_list[0].topk(1).indices
            predictions2.extend(prediction)

            predictions.extend(prediction)
            all_labels.extend(labels.tolist())

            if 1 in labels[prediction2]:
                accuracy2 += 1
            # correct = 0
            # for pred, label in zip(prediction, labels):
            #     if pred == label:
            #         correct += 1
            # accuracy += correct / len(labels)
            # else:
            # print(batches[0][0], batches[0][1])
            nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    return eval_loss.item(), accuracy / len(eval_dataloader), accuracy2 / len(eval_dataloader)
