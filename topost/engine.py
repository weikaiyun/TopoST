"""Forecasting MAE pretraining and zero-shot evaluation."""
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from .graph import pack_graphs


def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def forecasting_mae(prediction, target, threshold=0.0):
    valid = target > threshold
    errors = (prediction - target).abs()[valid]
    return errors.mean() if errors.numel() else prediction.sum() * 0


class SourceBatches(ConcatDataset):
    def __getitem__(self, index):
        dataset_index = int(np.searchsorted(self.cumulative_sizes, index, side='right'))
        local_index = index - (self.cumulative_sizes[dataset_index - 1] if dataset_index else 0)
        history, target, graph_index = self.datasets[dataset_index][local_index]
        return history, target, dataset_index, graph_index


def train_epoch(model, datasets, optimizer, device, max_grad_norm=5.0, mae_threshold=0.0, reconstruction_weight=0.1):
    model.train()
    loader = DataLoader(SourceBatches(datasets), batch_size=None, shuffle=True, num_workers=0)
    graph_inputs = [[graph.to(device) for graph in dataset.graphs] for dataset in datasets]
    total = 0.0
    for history, target, source_index, graph_index in tqdm(loader, desc='Pretrain', leave=False):
        history, target = history.to(device), target.to(device)
        prediction, auxiliary = model(history, graph_inputs[source_index][graph_index],
                                      target[..., 1:3], target.shape[1], return_aux=True)
        loss = forecasting_mae(prediction, target[..., :1], mae_threshold) + reconstruction_weight * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite forecasting loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


class ForecastMetrics:
    def __init__(self, mae_threshold=0.0, mape_threshold=0.001):
        self.mae_threshold, self.mape_threshold = mae_threshold, mape_threshold
        self.absolute = self.squared = self.percentage = 0.0
        self.count = self.percentage_count = 0

    def update(self, prediction, target):
        if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
            raise FloatingPointError('Non-finite prediction or target.')
        errors = prediction.double() - target.double()
        valid = target > self.mae_threshold
        self.absolute += errors[valid].abs().sum().item()
        self.squared += errors[valid].square().sum().item()
        self.count += valid.sum().item()
        valid_percentage = target > self.mape_threshold
        self.percentage += (errors[valid_percentage] / target[valid_percentage]).abs().sum().item()
        self.percentage_count += valid_percentage.sum().item()

    def compute(self):
        if not self.count or not self.percentage_count:
            raise ValueError('No valid evaluation targets.')
        return {'MAE': self.absolute / self.count,
                'RMSE': (self.squared / self.count) ** 0.5,
                'MAPE_percent': 100 * self.percentage / self.percentage_count}


@torch.no_grad()
def predict_graph(model, history, graphs, future_calendar, output_steps):
    """Scatter disjoint forecasts into the dataset's original node order."""
    prediction = history.new_empty((len(history), output_steps, history.shape[2], 1))
    for graph in graphs:
        prediction[:, :, graph.nodes] = model(history[:, :, graph.nodes], graph, future_calendar[:, :, graph.nodes], output_steps)
    return prediction


@torch.no_grad()
def evaluate(model, batches, graphs, device, mode='P', max_nodes=128,
             mae_threshold=0.0, mape_threshold=0.001):
    model.eval()
    if mode not in ('S', 'P'):
        raise ValueError('Inference mode must be S or P.')
    if mode == 'P':
        graphs = pack_graphs(graphs, max_nodes)
    graphs = [graph.to(device) for graph in graphs]
    metrics = ForecastMetrics(mae_threshold, mape_threshold)
    for history, target, _ in tqdm(DataLoader(batches, batch_size=None), desc='Evaluate', leave=False):
        history, target = history.to(device), target.to(device)
        prediction = predict_graph(model, history, graphs, target[..., 1:3], target.shape[1])
        metrics.update(prediction, target[..., :1])
    return metrics.compute()


def save_checkpoint(path, model, model_config, epoch):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({'model': model.state_dict(), 'model_config': model_config, 'epoch': epoch}, temporary)
    temporary.replace(path)
