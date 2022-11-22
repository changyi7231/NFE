import os
import time
import random
import datetime
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from model import Model
from datasets import KnowledgeGraph


def main(args):
    if args.mode == 'train':
        device = torch.device(args.device)
        save_dir = get_save_dir(args)

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
        dataset = KnowledgeGraph(file_path, args.dataset)
        model = Model(args.model_name, dataset.num_entity, dataset.num_relation, args.dimension, args.gamma).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.decay_rate)

        best_mrr = train(args, device, save_dir, dataset, model, optimizer, scheduler)
        return best_mrr

    elif args.mode == 'test':
        device = torch.device(args.device)
        save_dir = get_save_dir(args)

        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
        dataset = KnowledgeGraph(file_path, args.dataset)
        model = Model(args.model_name, dataset.num_entity, dataset.num_relation, args.dimension, args.gamma).to(device)

        state_file = os.path.join(save_dir, 'epoch_best.pth')
        if not os.path.isfile(state_file):
            raise RuntimeError('file {0} is not found'.format(state_file))
        print('load checkpoint {0}'.format(state_file))
        checkpoint = torch.load(state_file, device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        test(args, device, dataset, model, epoch, is_test=True)

    else:
        raise RuntimeError('wrong mode')


def train(args, device, save_dir, dataset, model, optimizer, scheduler):
    best_mrr = 0.0
    best_epoch = 0
    data = dataset.train_data
    number = len(data)
    batch_size = args.batch_size
    num_batch = len(data) // batch_size + int(len(data) % batch_size > 0)
    log = {'loss': [], 'train': {'loss': [], 'time': []}, 'valid': {'metric': [], 'time': []}}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.
        np.random.shuffle(data)
        model.train()
        for i in range(num_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(data))
            batch_data = data[start:end]
            heads = torch.LongTensor(batch_data[:, 0]).to(device)
            relations = torch.LongTensor(batch_data[:, 1]).to(device)
            tails = torch.LongTensor(batch_data[:, 2]).to(device)
            scores = model(heads, relations)

            mask = torch.zeros_like(scores)
            mask = mask.scatter(1, tails.unsqueeze(-1), 1)
            loss = torch.mean(F.softplus((1 - 2 * mask) * scores).sum(-1))

            total_loss += loss.item() * (end - start)
            log['loss'].append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss = total_loss / number
        t1 = time.time()
        print('\n[train: epoch {0}], loss: {1}, time: {2}s'.format(epoch, total_loss, t1 - t0))
        log['train']['loss'].append(total_loss)
        log['train']['time'].append(t1 - t0)

        if not (epoch % args.valid_interval):
            metric, t = test(args, device, dataset, model, epoch, is_test=False)
            log['valid']['metric'].append(metric)
            log['valid']['time'].append(t)
            mrr = metric['mrr']
            if mrr > best_mrr:
                best_mrr = mrr
                best_epoch = epoch
                save(save_dir, epoch, log, model)
    print('best mrr: {0} at epoch {1}'.format(best_mrr, best_epoch))
    return best_mrr


def test(args, device, dataset, model, epoch, is_test=True):
    if is_test:
        data = dataset.test_data
    else:
        data = dataset.valid_data
    number = len(data)
    batch_size = args.batch_size
    num_batch = number // batch_size + int(number % batch_size > 0)
    metric = {'mr': 0.0, 'mrr': 0.0, 'hit@1': 0.0, 'hit@3': 0.0, 'hit@10': 0.0}

    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for i in range(num_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, number)
            batch_data = data[start:end]
            heads = torch.LongTensor(batch_data[:, 0]).to(device)
            relations = torch.LongTensor(batch_data[:, 1]).to(device)
            scores = model(heads, relations)
            scores = scores.detach().cpu().numpy()

            for j in range(batch_data.shape[0]):
                target = scores[j, batch_data[j][2]]
                scores[j, dataset.hr_vocab[(batch_data[j][0], batch_data[j][1])]] = -1e8
                rank = np.sum(scores[j] >= target) + 1
                metric['mr'] += rank
                metric['mrr'] += (1.0 / rank)
                if rank == 1:
                    metric['hit@1'] += 1.0
                if rank <= 3:
                    metric['hit@3'] += 1.0
                if rank <= 10:
                    metric['hit@10'] += 1.0

    metric['mr'] /= number
    metric['mrr'] /= number
    metric['hit@1'] /= number
    metric['hit@3'] /= number
    metric['hit@10'] /= number
    t1 = time.time()
    print('[test: epoch {0}], mrr: {1}, mr: {2}, hit@1: {3}, hit@3: {4}, hit@10: {5}, time: {6}s'
          .format(epoch, metric['mrr'], metric['mr'], metric['hit@1'], metric['hit@3'], metric['hit@10'], t1-t0))
    return metric, t1 - t0


def get_save_dir(args):
    if args.save_dir:
        save_dir = args.save_dir
    else:
        name = str(datetime.datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'save', name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    return save_dir


def save(save_dir, epoch, log, model):
    state_path = os.path.join(save_dir, 'epoch_best.pth')
    state = {
        'epoch': epoch,
        'log': log,
        'model': model.state_dict(),
    }
    torch.save(state, state_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding by Normalizing Flows')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='mode')
    parser.add_argument('--device', type=str, default='cuda:0',
                        choices=['cuda:0', 'cpu'],
                        help='device')
    parser.add_argument('--dataset', type=str, default='WN18RR',
                        choices=['WN18', 'WN18RR', 'FB15k', 'FB15k-237', 'YAGO3-10'],
                        help='dataset')
    parser.add_argument('--model_name', type=str, default='NFE-1',
                        choices=['NFE-1', 'NFE-2', 'NFE-3', 'NFE-w/o-uncertainty', 'NFE-sigma', 'TransE', 'DistMult'],
                        help='model name')
    parser.add_argument('--save_dir', type=str, default='',
                        help='save directory')
    parser.add_argument('--valid_interval', type=int, default=1,
                        help='number of epochs to valid')

    parser.add_argument('--dimension', type=int, default=1024,
                        help='the dimension of each part')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--gamma', type=int, default=0,
                        help='gamma')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.0,
                        help='decay rate')

    parser.add_argument('--seed', type=int, default=0,
                        help='seed')
    parse_args = parser.parse_args()
    random.seed(parse_args.seed)
    np.random.seed(parse_args.seed)
    torch.manual_seed(parse_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(parse_args.seed)

    main(parse_args)
