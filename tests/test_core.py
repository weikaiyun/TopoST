import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from topost.data import baseline_ratio, calendar_features, dataset_calendar, chronological_split, TemporalBatches
from topost.engine import ForecastMetrics, forecasting_mae, predict_graph
from topost.graph import make_graph, pack_graphs, partition_graph, structural_descriptors
from topost.model import TopoST

ROOT = Path(__file__).resolve().parents[1]


class TopoSTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.adjacency = np.array([[0, 2, 0, 1, 0], [0, 0, 3, 0, 0], [1, 0, 0, 2, 0],
                                   [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float)
        self.graph = make_graph(self.adjacency, np.arange(5))

    def model(self, **kwargs):
        return TopoST(embed_dim=32, heads=4, gin_hidden_dim=16, depth=1, **kwargs)

    def sample(self, steps, ratio=False):
        calendar = torch.from_numpy(calendar_features(2 * steps, 1440 // steps, 5))
        calendar = calendar[None, :, None, :].expand(2, -1, 5, -1)
        parts = [torch.rand(2, steps, 5, 1) + 1, calendar[:, :steps]]
        if ratio:
            parts.append(torch.rand(2, steps, 5, 1) + 1)
        return torch.cat(parts, dim=-1), calendar[:, steps:]

    def test_equivariance_and_isolated_nodes(self):
        permutation = np.array([3, 0, 4, 1, 2])
        permuted = self.adjacency[np.ix_(permutation, permutation)]
        np.testing.assert_allclose(structural_descriptors(permuted),
                                   structural_descriptors(self.adjacency)[permutation], atol=1e-6)
        model = self.model().eval()
        history, future = self.sample(48)
        with torch.no_grad():
            original = model(history, self.graph, future)
            changed = model(history[:, :, permutation], make_graph(permuted, np.arange(5)), future[:, :, permutation])
        torch.testing.assert_close(changed, original[:, :, permutation], atol=2e-5, rtol=2e-5)

    def test_sequential_packed_inference_and_node_order(self):
        graphs = [make_graph(self.adjacency, [3, 0, 4]), make_graph(self.adjacency, [2, 1])]
        model = self.model().eval()
        for steps in (48, 288):
            history, future = self.sample(steps)
            sequential = predict_graph(model, history, graphs, future, steps)
            packed = predict_graph(model, history, pack_graphs(graphs), future, steps)
            torch.testing.assert_close(sequential, packed, atol=2e-5, rtol=2e-5)
            self.assertEqual(packed.shape, (2, steps, 5, 1))
        parts = partition_graph(self.adjacency, core_size=2)
        self.assertEqual(sorted(torch.cat([g.nodes for g in parts]).tolist()), list(range(5)))

    def test_optional_paths_and_gradients(self):
        for ratio, reconstruction in itertools.product((False, True), repeat=2):
            model = self.model(use_baseline_ratio=ratio, use_reconstruction=reconstruction,
                               reconstruction_mask_ratio=0.5).train()
            history, future = self.sample(48, ratio)
            target = torch.rand(2, 48, 5, 1) + 1
            prediction, auxiliary = model(history, self.graph, future, return_aux=True)
            loss = forecasting_mae(prediction, target) + 0.1 * auxiliary
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertEqual(model.embedding.projection.in_features, 12 * (1 + int(ratio)))
            self.assertEqual(model.reconstruction_head is not None, reconstruction)
            self.assertEqual(auxiliary.item() > 0, reconstruction)

    def test_causal_ratio_and_target_channels(self):
        flow = np.arange(1, 41, dtype=np.float32).reshape(20, 2)
        ratio = baseline_ratio(flow, steps_per_day=4, days=2)
        np.testing.assert_array_equal(ratio[:4], np.ones((4, 2)))
        expected = flow[8] / ((flow[4] + flow[0]) / 2 + 1e-6)
        np.testing.assert_allclose(ratio[8], expected)
        changed = flow.copy()
        changed[12:] *= 50
        np.testing.assert_array_equal(baseline_ratio(changed, 4, 2)[:12], ratio[:12])
        calendar = np.broadcast_to(calendar_features(20, 30, 1)[:, None], (20, 2, 2))
        channels = np.concatenate([flow[..., None], calendar, ratio[..., None]], axis=-1)
        batches = TemporalBatches(channels, steps=3, batch_size=4)
        history, target, _ = batches[0]
        self.assertEqual(history.shape[-1], 4)
        self.assertEqual(target.shape[-1], 3)
        np.testing.assert_array_equal(target[0, ..., 0], flow[3:6])
        np.testing.assert_array_equal(target[0, ..., 1:3], calendar[3:6])

    def test_calendar_boundaries_and_known_future_inputs(self):
        five_minute = calendar_features(289, 5, 7)
        np.testing.assert_array_equal(five_minute[[0, 287, 288]], [[5, 7], [1440, 7], [5, 1]])
        weekdays = calendar_features(289, 5, 5, weekday_only=True)
        np.testing.assert_array_equal(weekdays[288], [5, 1])
        shanghai = dataset_calendar('TrafficSH', 14, 30)
        np.testing.assert_array_equal(shanghai[[0, 11, 12]], [[1110, 1], [1440, 1], [30, 2]])
        self.assertTrue((shanghai > 0).all())
        model = self.model().eval()
        history, future = self.sample(48)
        changed = future.clone()
        changed[..., 0] = (changed[..., 0] - 1 + 120) % 1440 + 1
        with torch.no_grad():
            original = model(history, self.graph, future)
            shifted = model(history, self.graph, changed)
        self.assertFalse(torch.allclose(original, shifted))
        invalid = torch.cat([torch.zeros_like(future[..., :1]), future], dim=-1)
        with self.assertRaisesRegex(ValueError, 'future minute/weekday'):
            model(history, self.graph, invalid)

    def test_split_boundaries_tail_windows_and_global_metrics(self):
        data = np.arange(41, dtype=np.float32).reshape(41, 1, 1)
        train, validation, test = chronological_split(data, 0.2, 0.6)
        np.testing.assert_array_equal(np.concatenate([train, validation, test]), data)
        batches = TemporalBatches(test, steps=3, batch_size=5)
        self.assertEqual(len(batches.starts), len(test) - 6 + 1)
        self.assertEqual(batches[0][0][0, 0, 0, 0].item(), test[0, 0, 0])
        self.assertEqual(batches[len(batches) - 1][1][-1, -1, 0, 0].item(), test[-1, 0, 0])
        metrics = ForecastMetrics()
        metrics.update(torch.tensor([1., 5., 99.]), torch.tensor([2., 3., 0.]))
        metrics.update(torch.tensor([8.]), torch.tensor([4.]))
        result = metrics.compute()
        self.assertAlmostEqual(result['MAE'], 7 / 3)
        self.assertAlmostEqual(result['RMSE'], math.sqrt(21 / 3))
        self.assertAlmostEqual(result['MAPE_percent'], (0.5 + 2 / 3 + 1) / 3 * 100)

    def test_pretrain_save_reload_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = np.random.default_rng(3)
            for name, shape in [('PEMS08', (608, 8, 1)), ('CHI_TAXI', (8, 240, 1))]:
                dataset = root / 'data' / name
                dataset.mkdir(parents=True)
                np.savez(dataset / f'{name}.npz', data=generator.uniform(1, 10, shape).astype(np.float32))
                adjacency = np.roll(np.eye(8), 1, axis=0)
                if name == 'PEMS08':
                    edges = 'from,to,cost\n' + ''.join(f'{i},{(i+1)%8},1\n' for i in range(8))
                    (dataset / f'{name}.csv').write_text(edges)
                else:
                    np.save(dataset / f'{name}_rn_adj.npy', adjacency)
            config = json.loads((ROOT / 'configs/topost.json').read_text())
            config['model'].update(embed_dim=32, heads=4, gin_hidden_dim=16, depth=1)
            config['partition']['core_size'] = 4
            config_path = root / 'topost.json'
            environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
            for enabled in (False, True):
                config['model'].update(use_baseline_ratio=enabled, use_reconstruction=enabled)
                config_path.write_text(json.dumps(config))
                output = root / f'weights_{enabled}'
                common = ['--config', str(config_path), '--data-root', str(root / 'data'), '--device', 'cpu']
                commands = [
                    [sys.executable, str(ROOT / 'train.py'), *common, '--datasets', 'PEMS08',
                     '--epochs', '1', '--output', str(output)],
                    [sys.executable, str(ROOT / 'evaluate.py'), *common, '--datasets', 'CHI_TAXI',
                     '--checkpoint', str(output / 'final.pt')],
                ]
                for command in commands:
                    result = subprocess.run(command, cwd=root, env=environment, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stdout[-6000:])
                self.assertIn('CHI_TAXI,', result.stdout)
                final = torch.load(output / 'final.pt', weights_only=True)
                epoch = torch.load(output / 'epoch_1.pt', weights_only=True)
                self.assertEqual(final['epoch'], 1)
                for key in final['model']:
                    torch.testing.assert_close(final['model'][key], epoch['model'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
