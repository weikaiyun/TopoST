"""Pretrain TopoST jointly on the twenty source datasets."""
import argparse
import json
from pathlib import Path
import torch

from topost.data import SOURCES, TemporalBatches, load_adjacency, load_series
from topost.engine import seed_everything, train_epoch, save_checkpoint
from topost.graph import partition_graph
from topost.model import TopoST

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/topost.json')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--datasets', help='Comma-separated subset of source datasets.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    options = config['training']
    epochs = options['epochs'] if args.epochs is None else args.epochs
    names = args.datasets.split(',') if args.datasets else config['source_datasets']
    if epochs < 1 or not names or len(set(names)) != len(names) or any(n not in SOURCES for n in names):
        parser.error('Expected positive epochs and distinct source datasets.')
    if args.dry_run:
        print(f'Sources ({len(names)}): {", ".join(names)}')
        print(f'Epochs: {epochs}; batch: {options["batch_size"]}; seed: {options["seed"]}')
        print(f'Final checkpoint: {args.output / "final.pt"}')
        return
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new --output directory.')
    seed_everything(options['seed'])
    device = torch.device(args.device)
    datasets = []
    for name in names:
        data, interval = load_series(args.data_root, name, config['model']['use_baseline_ratio'],
                                     config['data']['baseline_days'])
        adjacency = load_adjacency(args.data_root, name, data.shape[1])
        graphs = partition_graph(adjacency, **config['partition'])
        dataset = TemporalBatches(data, 24 * 60 // interval, options['batch_size'],
                                  graphs, seed=options['seed'])
        datasets.append(dataset)
        print(f'{name}: {data.shape[0]} steps, {data.shape[1]} nodes, {len(graphs)} units, {len(dataset)} batches')
    model = TopoST(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options['learning_rate'])
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, datasets, optimizer, device, options['max_grad_norm'],
                           config['metrics']['mae_threshold'], options['reconstruction_weight'])
        save_checkpoint(args.output / f'epoch_{epoch}.pt', model, config['model'], epoch)
        print(f'Epoch {epoch}/{epochs}: loss = {loss:.6f}', flush=True)
    save_checkpoint(args.output / 'final.pt', model, config['model'], epochs)
    print(f'Final checkpoint: {args.output / "final.pt"}')


if __name__ == '__main__':
    main()
