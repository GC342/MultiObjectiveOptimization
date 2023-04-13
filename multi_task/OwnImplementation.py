import torch
import torch.optim as optim
import json
import datetime
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from timeit import default_timer as timer
import click

import datasets
import losses
import metrics
import model_selector
from min_norm_solvers import MinNormSolver, gradient_normalizers

NUM_EPOCHS = 100

@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')
def train_multi_task(param_file):
    params, configs = load_params(param_file)
    writer = setup_experiment(params)
    train_loader, train_dst, val_loader, val_dst, loss_fn, metric, model = prepare_data_and_model(params, configs)
    optimizer = create_optimizer(params, model)

    main_training_loop(params, configs, writer, train_loader, val_loader, model, loss_fn, metric, optimizer)
    writer.close()


def load_params(param_file):
    with open('configs.json') as config_params:
        configs = json.load(config_params)

    with open(param_file) as json_params:
        params = json.load(json_params)

    return params, configs


def setup_experiment(params):
    exp_identifier = [f'{key}={val}' for key, val in params.items() if key != 'tasks']
    exp_identifier = ','.join(exp_identifier)
    params['exp_id'] = exp_identifier

    return SummaryWriter(log_dir='runs/{}'.format(datetime.datetime.now().strftime("%I.%M%p on %B %d, %Y")))


def prepare_data_and_model(params, configs):
    train_loader, train_dst, val_loader, val_dst = datasets.get_dataset(params, configs)
    loss_fn = losses.get_loss(params)
    metric = metrics.get_metrics(params)

    model = model_selector.get_model(params)

    return train_loader, train_dst, val_loader, val_dst, loss_fn, metric, model


def create_optimizer(params, model):
    model_params = []
    for m in model.values():
        model_params += list(m.parameters())

    optimizer_type = params['optimizer']
    lr = params['lr']

    if optimizer_type == 'RMSprop':
        return optim.RMSprop(model_params, lr=lr)
    elif optimizer_type == 'Adam':
        return optim.Adam(model_params, lr=lr)
    elif optimizer_type == 'SGD':
        return optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_type}')


def main_training_loop(params, configs, writer, train_loader, val_loader, model, loss_fn, metric, optimizer):
    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']

    for epoch in range(NUM_EPOCHS):
        start = timer()

        # Update learning rate every 10 epochs
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.85

        # Train the model
        train_single_epoch(epoch, tasks, all_tasks, train_loader, model, loss_fn, optimizer, params, writer)

        # Evaluate the model
        evaluate_model(epoch, tasks, all_tasks, val_loader, model, loss_fn, metric, writer)

        # Save the model every 3 epochs
        if epoch % 3 == 0:
            save_model(params, epoch, model, optimizer)

        end = timer()
        print(f'Epoch {epoch} ended in {end - start:.2f}s')

def train_single_epoch(epoch, tasks, all_tasks, train_loader, model, loss_fn, optimizer, params, writer):
    print(f'Epoch {epoch} Started')

    for m in model.values():
        m.train()

    n_iter = 0
    for batch in train_loader:
        n_iter += 1
        images = Variable(batch[0].cuda())

        labels = {t: Variable(batch[i + 1].cuda()) for i, t in enumerate(tasks)}
        logits = {t: model[t](images) for t in tasks}

        losses = {t: loss_fn[t](logits[t], labels[t]) for t in tasks}
        total_loss = sum(losses.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalars('train/loss', {t: losses[t].item() for t in tasks}, n_iter)

    print(f'Epoch {epoch} Training Loss: {total_loss.item():.4f}')


def evaluate_model(epoch, tasks, all_tasks, val_loader, model, loss_fn, metric, writer):
    print(f'Epoch {epoch} Evaluation Started')

    for m in model.values():
        m.eval()

    n_iter = 0
    with torch.no_grad():
        for batch in val_loader:
            n_iter += 1
            images = Variable(batch[0].cuda())
            labels = {t: Variable(batch[i + 1].cuda()) for i, t in enumerate(tasks)}
            logits = {t: model[t](images) for t in tasks}

            losses = {t: loss_fn[t](logits[t], labels[t]) for t in tasks}
            total_loss = sum(losses.values())

            metrics_scores = {t: metric[t](logits[t], labels[t]) for t in tasks}

            writer.add_scalars('val/loss', {t: losses[t].item() for t in tasks}, n_iter)
            writer.add_scalars('val/metric', {t: metrics_scores[t].item() for t in tasks}, n_iter)

    print(f'Epoch {epoch} Evaluation Loss: {total_loss.item():.4f}')


def save_model(params, epoch, model, optimizer):
    save_path = f'saved_models/{params["exp_id"]}_epoch_{epoch}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': {t: model[t].state_dict() for t in model},
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f'Model saved at {save_path}')


if __name__ == '__main__':
    train_multi_task()







