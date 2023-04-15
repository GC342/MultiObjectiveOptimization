import torch
import torch.optim as optim
import json
import datetime
from tqdm import tqdm
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

    main_training_loop(params, configs, writer, train_loader, val_loader, val_dst, model, loss_fn, metric, optimizer)
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


def main_training_loop(params, configs, writer, train_loader, val_loader, val_dst, model, loss_fn, metric, optimizer):
    tasks = params['tasks']
    all_tasks = configs[params['dataset']]['all_tasks']

    for epoch in tqdm(range(NUM_EPOCHS)):
        start = timer()

        # Update learning rate every 10 epochs
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.85

        # Train the model
        train_single_epoch(epoch, tasks, train_loader, model, loss_fn, optimizer, params, writer)

        # Evaluate the model
        validate_model(epoch, tasks, val_loader, val_dst, model, loss_fn, metric, writer)

        # Save the model every 3 epochs
        if epoch % 3 == 0:
            save_model(params, epoch, model, optimizer)

        end = timer()
        print(f'Epoch {epoch} ended in {end - start:.2f}s')


def train_single_epoch(epoch, tasks, train_loader, model, loss_fn, optimizer, params, writer):
    print(f'Epoch {epoch} Started')

    mask = None
    masks = {}
    loss_data = {}
    grads = {}
    scale = {}

    for m in model.values():
        m.train()

    n_iter = 0
    for batch in train_loader:
        n_iter += 1
        images = Variable(batch[0].cuda())

        labels = {t: Variable(batch[i + 1].cuda()) for i, t in enumerate(tasks)}

        with torch.no_grad():
            images_no_grad = Variable(images.data)

        rep, mask = model['rep'](images_no_grad, mask)

        if isinstance(rep, list):
            # This is a hack to handle psp-net
            rep = rep[0]
            rep_variable = [Variable(rep.data.clone(), requires_grad=True)]
            list_rep = True
        else:
            rep_variable = Variable(rep.data.clone(), requires_grad=True)
            list_rep = False

        for t in tasks:
            optimizer.zero_grad()
            out_t, masks[t] = model[t](rep_variable, None)
            loss = loss_fn[t](out_t, labels[t])
            loss_data[t] = loss.item()
            loss.backward()
            grads[t] = []

            if list_rep:
                grads[t].append(Variable(rep_variable[0].grad.data.clone(), requires_grad=False))
                rep_variable[0].grad.data.zero_()
            else:
                grads[t].append(Variable(rep_variable.grad.data.clone(), requires_grad=False))
                rep_variable.grad.data.zero_()

        gn = gradient_normalizers(grads, loss_data, params['normalization_type'])
        for t in tasks:
            for gr_i in range(len(grads[t])):
                grads[t][gr_i] = grads[t][gr_i] / gn[t]

        # Frank-Wolfe iteration to compute scales.
        sol, min_norm = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
        for i, t in enumerate(tasks):
            scale[t] = float(sol[i])

        optimizer.zero_grad()
        rep, _ = model['rep'](images, mask)
        for i, t in enumerate(tasks):
            out_t, _ = model[t](rep, masks[t])
            loss_t = loss_fn[t](out_t, labels[t])
            loss_data[t] = loss_t.item()
            if i > 0:
                loss = loss + scale[t] * loss_t
            else:
                loss = scale[t] * loss_t
        loss.backward()
        optimizer.step()

        writer.add_scalar('training_loss', loss.item(), n_iter)
        for t in tasks:
            writer.add_scalar('training_loss_{}'.format(t), loss_data[t], n_iter)

    for m in model:
        model[m].eval()




def validate_model(epoch, tasks, val_loader, val_dst, model, loss_fn, metric, writer):
    print(f'Epoch {epoch} Evaluation Started')
    n_iter = 0

    for m in model:
        model[m].eval()

    total_loss = {}
    total_loss['all'] = 0.0
    met = {}
    for t in tasks:
        total_loss[t] = 0.0
        met[t] = 0.0

    num_val_batches = 0
    for batch_val in val_loader:
        n_iter += 1
        val_images = Variable(batch_val[0].cuda(), volatile=True)
        labels_val = {}

        labels_val = {t: Variable(batch_val[i + 1].cuda()) for i, t in enumerate(tasks)}

        val_rep, _ = model['rep'](val_images, None)
        for t in tasks:
            out_t_val, _ = model[t](val_rep, None)
            loss_t = loss_fn[t](out_t_val, labels_val[t])
            total_loss['all'] += loss_t.data[0]
            total_loss[t] += loss_t.data[0]
            metric[t].update(out_t_val, labels_val[t])
        num_val_batches += 1

    for t in tasks:
        writer.add_scalar('validation_loss_{}'.format(t), total_loss[t] / num_val_batches, n_iter)
        metric_results = metric[t].get_result()
        for metric_key in metric_results:
            writer.add_scalar('metric_{}_{}'.format(metric_key, t), metric_results[metric_key], n_iter)
        metric[t].reset()
    writer.add_scalar('validation_loss', total_loss['all'] / len(val_dst), n_iter)


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







