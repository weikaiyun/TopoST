"""OpenCity benchmark periods and lazy temporal mini-batches."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

SOURCES = (
    'CAD4-1', 'CAD4-2', 'CAD4-3', 'CAD4-4', 'CAD7-1', 'CAD7-2', 'CAD7-3',
    'CAD8-1', 'CAD8-2', 'CAD12-1', 'CAD12-2', 'PEMS08', 'PEMS04',
    'PEMS_BAY', 'METR_LA', 'NYC_TAXI', 'TrafficHZ', 'TrafficZZ', 'TrafficCD', 'TrafficJN',
)
TARGETS = ('CAD3', 'CAD5', 'PEMS07M', 'TrafficSH', 'CHI_TAXI', 'NYC_BIKE-3')


def baseline_ratio(flow, steps_per_day, days=7):
    """Causal same-slot historical mean; the first day has neutral ratio one."""
    if days < 1 or steps_per_day < 1:
        raise ValueError('History days and daily sampling steps must be positive.')
    sums = np.zeros_like(flow, dtype=np.float64)
    counts = np.zeros((len(flow), 1), dtype=np.float64)
    for day in range(1, days + 1):
        lag = day * steps_per_day
        if lag >= len(flow):
            break
        sums[lag:] += flow[:-lag]
        counts[lag:] += 1
    means = sums / np.maximum(counts, 1)
    ratio = np.divide(flow, means + 1e-6, out=np.ones_like(flow, dtype=np.float64), where=counts > 0)
    return np.clip(ratio, 0.1, 10).astype(np.float32)



def calendar_features(length, interval, week_start, day_start=0, weekday_only=False):
    """Minute/weekday indices aligned to the selected benchmark sequence.

    Minutes label interval ends (5..1440 or 30..1440); weekdays are 1..7.
    PEMS07M omits weekends and cycles through weekdays 1..5.
    """
    if interval < 1 or 1440 % interval or day_start % interval:
        raise ValueError('Calendar sampling must divide a day into whole intervals.')
    week_length = 5 if weekday_only else 7
    if not 1 <= week_start <= week_length or not 0 <= day_start < 1440:
        raise ValueError('Invalid initial weekday or minute-of-day.')
    slots_per_day = 1440 // interval
    slots = np.arange(length, dtype=np.int64) + day_start // interval
    minute = (slots % slots_per_day + 1) * interval
    weekday = (week_start - 1 + slots // slots_per_day) % week_length + 1
    return np.stack([minute, weekday], axis=-1).astype(np.float32)


def dataset_calendar(name, length, interval):
    if name.startswith('CAD'):
        week_start = 3 if '-' in name else 7
    else:
        week_start = {'PEMS04': 1, 'PEMS08': 5, 'METR_LA': 4, 'PEMS_BAY': 7,
                      'PEMS07M': 2, 'NYC_TAXI': 5, 'CHI_TAXI': 5,
                      'NYC_BIKE-3': 3}.get(name, 1)
    day_start = 18 * 60 if name.startswith('Traffic') else 0
    return calendar_features(length, interval, week_start, day_start,
                             weekday_only=name == 'PEMS07M')

def load_series(root, name, use_baseline_ratio=False, baseline_days=7):
    if name not in SOURCES + TARGETS:
        raise ValueError(f'Unknown Table 1 dataset: {name}')
    with np.load(Path(root) / name / f'{name}.npz', allow_pickle=False) as archive:
        data = archive['data']
    interval = 30 if name in ('NYC_TAXI', 'CHI_TAXI', 'NYC_BIKE-3') or name.startswith('Traffic') else 5
    if name in ('NYC_TAXI', 'CHI_TAXI'):
        data = data[:, :, 0].T
    elif data.ndim == 3:
        data = data[:, :, 0]
    if name == 'NYC_TAXI':
        data = data[:(366 + 365 + 365 + 365 + 366) * 48]
    elif name == 'NYC_BIKE-3':
        data = data[(366 + 365 + 365 + 365) * 48:(366 + 365 + 365 + 365 + 366) * 48]
    elif name.startswith('CAD'):
        data = data[:60 * 288] if '-' in name else data[60 * 288:121 * 288]
    elif name == 'METR_LA':
        data = data[:61 * 288]
    elif name == 'PEMS_BAY':
        data = data[:59 * 288]
    data = np.asarray(data, dtype=np.float32).copy()
    data[np.isnan(data)] = 0
    if data.ndim != 2 or not data.size or not np.isfinite(data).all():
        raise ValueError(f'{name}: expected finite nonempty [time, node] measurements.')
    calendar = dataset_calendar(name, len(data), interval)
    calendar = np.broadcast_to(calendar[:, None, :], (len(data), data.shape[1], 2))
    channels = [data[:, :, None], calendar]
    if use_baseline_ratio:
        ratio = baseline_ratio(data, 24 * 60 // interval, baseline_days)
        channels.append(ratio[:, :, None])
    return np.concatenate(channels, axis=-1), interval


def load_adjacency(root, name, num_nodes):
    directory = Path(root) / name
    if name in ('PEMS04', 'PEMS08'):
        adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        with (directory / f'{name}.csv').open(newline='') as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                source, target = int(row[0]), int(row[1])
                if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                    raise ValueError(f'{name}: edge endpoint outside the node range.')
                adjacency[source, target] = 1
    else:
        adjacency = np.load(directory / f'{name}_rn_adj.npy', allow_pickle=False)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(f'{name}: graph and measurements have different node counts.')
    return adjacency


def chronological_split(data, val_ratio, test_ratio):
    if not (0 <= val_ratio <= 1 and 0 <= test_ratio <= 1 and val_ratio + test_ratio <= 1):
        raise ValueError('Invalid chronological split ratios.')
    train_end = len(data) - int(len(data) * (val_ratio + test_ratio))
    test_start = len(data) - int(len(data) * test_ratio)
    return data[:train_end], data[train_end:test_start], data[test_start:]


def target_test_split(data, name):
    val_ratio, test_ratio = (0.2, 0.6) if name in ('CHI_TAXI', 'NYC_BIKE-3') else (0.1, 0.4)
    return chronological_split(data, val_ratio, test_ratio)[2]


class TemporalBatches(Dataset):
    """One item is a complete temporal batch on a single graph."""
    def __init__(self, data, steps, batch_size, graphs=None, seed=42):
        if steps < 1 or batch_size < 1:
            raise ValueError('Window and batch sizes must be positive.')
        self.data, self.steps, self.batch_size = data, steps, batch_size
        self.graphs = graphs
        self.starts = np.arange(max(0, len(data) - 2 * steps + 1))
        if graphs is not None:
            np.random.default_rng(seed).shuffle(self.starts)
            self.starts = self.starts[:len(self.starts) // batch_size * batch_size]
        self.batches = (len(self.starts) + batch_size - 1) // batch_size
        if not self.batches:
            raise ValueError('The selected sequence cannot supply a complete forecasting batch.')

    def __len__(self):
        return self.batches * (len(self.graphs) if self.graphs is not None else 1)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        batch_index, graph_index = divmod(index, len(self.graphs)) if self.graphs is not None else (index, 0)
        starts = self.starts[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        nodes = self.graphs[graph_index].nodes.numpy() if self.graphs is not None else np.arange(self.data.shape[1])
        history = np.stack([self.data[start:start + self.steps, nodes] for start in starts])
        target = np.stack([self.data[start + self.steps:start + 2 * self.steps, nodes, :3] for start in starts])
        return torch.from_numpy(history), torch.from_numpy(target), graph_index
