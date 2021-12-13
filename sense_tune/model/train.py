import torch
from torch import optim
from tqdm import tqdm

from sense_tune.model.bert import BertSense, forward


def train(model, train_dataloader, device, learning_rate=1e-4,
          num_epochs=5, max_steps=10, max_grad_norm=1.0):
    """ Fine-tune the model """
    if max_steps > 0:
        max_steps
        num_epochs = max_steps // len(train_dataloader) + 1

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()  # torch.nn.BCEWithLogitsLoss()
    bin_loss_function = torch.nn.MSELoss()
    global_step = 0
    tr_loss, tr2_loss = 0.0, 0.0

    model.zero_grad()

    for epoch in range(num_epochs):

        # set_seed(args)  # Added here for reproducibility
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        # import pdb; pdb.set_trace()
        for step, batches in enumerate(epoch_iterator):
            model.train()

            # run model
            batch_loss = 0
            logits_list = []

            if len(batches) < 1:
                continue

            if isinstance(model, BertSense):
                for batch in batches:
                    logits = forward(model, batch, device)

                    targets = torch.max(batch[3].to(device), -1).indices.to(device).detach()
                    # batch_loss += loss_function(logits, targets)#, batch[3].to(device).detach())
                    batch_loss += loss_function(logits.unsqueeze(dim=0), targets.unsqueeze(dim=-1))

                    # logits = model.sigmoid(logits)
                    batch_loss += bin_loss_function(logits, batch[3].to(torch.float).to(device))
            else:
                batch = torch.stack([b[0] for b in batches])
                labels = torch.stack([l[1] for l in batches])

                logits = model(batch.to(device))
                targets = torch.max(labels.view(1, -1), -1).indices.to(device).detach()

                batch_loss += loss_function(logits.view(1, -1), targets)
                # batch_loss += bin_loss_function(logits.view(1, -1), labels.to(torch.float).to(device))

            logits_list.append(logits)

            loss = batch_loss / len(batches)
            # loss2 = bin_loss / len(batches)

            loss.backward()
            # loss2.backward()

            tr_loss += loss.item()
            # tr2_loss += loss2.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            model.zero_grad()
            global_step += 1

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < max_steps < global_step:
            break

    return global_step, tr_loss / global_step


def evaluate(model, eval_dataloader, device):
    eval_loss = 0.0
    nb_eval_steps = 0
    accuracy = 0

    loss_function = torch.nn.CrossEntropyLoss()

    predictions = []
    iterator = tqdm(eval_dataloader, desc="Iteration")
    for batches in iterator:
        if len(batches) < 1:
            continue

        model.eval()
        with torch.no_grad():

            labels = batches[0][3] if isinstance(model, BertSense) else [b[1] for b in batches]

            # run model
            batch_loss = 0
            logits_list = []

            if isinstance(model, BertSense):
                for batch in batches:
                    logits = forward(model, batch, device)
                    targets = torch.max(batch[3].to(device), -1).indices.to(device).detach()
                    # batch_loss += loss_function(logits, targets)#, batch[3].to(device).detach())
                    batch_loss += loss_function(logits.unsqueeze(dim=0), targets.unsqueeze(dim=-1))
            else:
                batch = torch.stack([b[0] for b in batches])
                lab = torch.stack([l[1] for l in batches])
                logits = model(batch.to(device)).view(1, -1)
                targets = torch.max(lab.view(1, -1), -1).indices.to(device).detach()

                batch_loss += loss_function(logits, targets)

            logits_list.append(logits)

            loss = batch_loss / len(batches)

        eval_loss += loss
        prediction = logits_list[0].topk(1).indices
        predictions.extend(prediction)

        print(prediction)
        print(labels)

        if 1 in labels[prediction]:
            accuracy += 1
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    return eval_loss.item(), accuracy / len(eval_dataloader)
