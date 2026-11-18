"""Evaluate the final pretrained checkpoint on the six unseen target datasets."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch

from topost.data import TARGETS, TemporalBatches, load_adjacency, load_series, target_test_split
from topost.engine import evaluate, seed_everything
from topost.graph import partition_graph
from topost.model import TopoST

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/topost.json')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/final.pt')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--datasets', help='Comma-separated subset of target datasets.')
    parser.add_argument('--mode', choices=('S', 'P'), default='P')
    parser.add_argument('--max-nodes', type=int, default=128)
    parser.add_argument('--output', type=Path, help='Optionally save the metrics as CSV.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    names = args.datasets.split(',') if args.datasets else config['target_datasets']
    if not names or len(set(names)) != len(names) or any(n not in TARGETS for n in names):
        parser.error('Select distinct Table 1 target datasets.')
    if args.max_nodes < 0:
        parser.error('--max-nodes must be nonnegative.')
    if args.dry_run:
        print(f'Targets ({len(names)}): {", ".join(names)}')
        print(f'Checkpoint: {args.checkpoint}; inference: TopoST-{args.mode}')
        return
    seed_everything(config['training']['seed'])
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = TopoST(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.requires_grad_(False)
    rows = []
    for name in names:
        data, interval = load_series(args.data_root, name, checkpoint['model_config']['use_baseline_ratio'],
                                     config['data']['baseline_days'])
        adjacency = load_adjacency(args.data_root, name, data.shape[1])
        graphs = partition_graph(adjacency, **config['partition'])
        batches = TemporalBatches(target_test_split(data, name), 24 * 60 // interval,
                                  config['training']['batch_size'])
        metrics = evaluate(model, batches, graphs, device, args.mode, args.max_nodes, **config['metrics'])
        rows.append({'dataset': name, **metrics})
    fields = ['dataset', 'MAE', 'RMSE', 'MAPE_percent']
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
