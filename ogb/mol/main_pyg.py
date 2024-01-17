import hashlib
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN
import expander

from tqdm import tqdm
import argparse
import time
import numpy as np
import os

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def make_experiment_dataset_dir(dataset_name: str, expander_config_hash: str):
    # we can't allow for experiments with different expander configurations to share the same pre-processed dataset.
    # this is why we split the same dataset into different directories based on the expander configuration.
    root_dir = 'dataset/' + expander_config_hash
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    
    base_root_dir = 'dataset/base'
    if not os.path.exists(base_root_dir):
        os.makedirs(base_root_dir)
    ds = PygGraphPropPredDataset(name = dataset_name, root='dataset/base')
    
    dataset_dir = root_dir + '/' + ds.dir_name
    dataset_base_dir = base_root_dir + '/' + ds.dir_name
    if not os.path.exists(dataset_dir):
        print('creating dataset dir: ' + dataset_dir)
        os.makedirs(dataset_dir)
        # copy all files from base dataset to experiment dataset (except for processed files)
        for filename in os.listdir(dataset_base_dir):
            if not filename.startswith('processed'):
                if os.path.isdir(dataset_base_dir + '/' + filename):
                    os.system('cp -r ' + dataset_base_dir + '/' + filename + ' ' + dataset_dir)
                else:
                    os.system('cp ' + dataset_base_dir + '/' + filename + ' ' + dataset_dir)
            
    return root_dir


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--expander', type=str, default='none', choices=expander.ExpanderConfig.get_preset_names())
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    expander_config = expander.ExpanderConfig.get_preset(args.expander)
    root_dir = make_experiment_dataset_dir(args.dataset, expander_config.hash())
    dataset = PygGraphPropPredDataset(name=args.dataset, root=root_dir, pre_transform=expander.PreTransform(expander_config))

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False, expander_config=expander_config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    save_file_name = f"{results_dir}/{hashlib.md5(str(vars(args)).encode('utf-8')).hexdigest()}.pt"

    counter = 1
    while os.path.exists(save_file_name):
        save_file_name = f"{results_dir}/{hashlib.md5(str(vars(args)).encode('utf-8')).hexdigest()}({counter}).pt"
        counter += 1
        if counter > 100:
            raise RuntimeError('Too many files with the same hash!')

    torch.save({
        'best_val_epoch': best_val_epoch, 
        'val_curve': valid_curve,
        'test_curve': test_curve,
        'train_curve': train_curve,
        'Val': valid_curve[best_val_epoch], 
        'Test': test_curve[best_val_epoch], 
        'Train': train_curve[best_val_epoch], 
        'BestTrain': best_train,
        'args': vars(args),
    }, save_file_name)


if __name__ == "__main__":
    main()
