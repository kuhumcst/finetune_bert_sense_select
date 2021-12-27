import torch


def save_checkpoint(path, model, valid_loss):
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, path)
    print(f'Saved model to: {path}')


def load_checkpoint(path, model, device):
    state_dict = torch.load(path, map_location=device)
    print(f'Loaded model from: {path}')
    print(state_dict.keys())
    model.load_state_dict(state_dict['model_state_dict'])
    return model, state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, path)
    print(f'Saved model to: {path}')


def load_metrics(path, device):
    state_dict = torch.load(path, map_location=device)
    print(f'Loaded model from: {path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']