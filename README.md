# TopoST

Pretraining and zero-shot evaluation for the OpenCity benchmark in Table 1.

## Setup

```bash
python -m pip install -r requirements.txt
```

Place the prepared OpenCity benchmark datasets under `data/`, or supply
`--data-root /path/to/data`. Each dataset directory contains:

```text
data/<dataset>/<dataset>.npz          # NumPy archive with key "data"
data/<dataset>/<dataset>_rn_adj.npy   # Spatial adjacency in measurement node order
```

PEMS04 and PEMS08 use `<dataset>.csv` with `from,to,cost` columns for their
spatial graph. Taxi arrays use `[node, time, channel]`; other arrays use
`[time, node]` or `[time, node, channel]`. Use the original benchmark arrays;
`topost/data.py` selects the source and target benchmark periods.
The source and target dataset names are listed in `configs/topost.json`.

## Pretraining and evaluation

```bash
bash scripts/pretrain.sh --data-root /path/to/data --device cuda:0
bash scripts/eval.sh --data-root /path/to/data --device cuda:0 --output results.csv
```

The default configuration trains on all 20 source datasets for five epochs,
with batch size 16 and Adam learning rate 0.001. Each training batch contains
temporal windows from one METIS unit. Completed epochs are saved as
`checkpoints/epoch_1.pt` through `epoch_5.pt`; `checkpoints/final.pt` contains
the final epoch used by evaluation. Use `--output` on the pretraining command
to select a fresh checkpoint directory, and `--checkpoint` on the evaluation
command to select its `final.pt`.

Evaluation uses six unseen target datasets without parameter updates. The
chronological train/validation/test splits are 50%/10%/40% for flow and speed,
and 20%/20%/60% for taxi and bike demand. Windows stay within their split.
Both history and forecast cover 24 hours: 288 steps at five-minute resolution
or 48 steps at thirty-minute resolution. Forecasts are restored to the
original node order. Historical and future calendar indices are constructed
from the benchmark time axis; future measurements are used only as evaluation
targets. MAE and RMSE include targets greater than zero; MAPE
includes targets greater than 0.001 and is reported in percent.

`--mode S` evaluates units sequentially. The default `--mode P --max-nodes 128`
packs complete units with block-diagonal graphs and masked adaptive adjacency.
Use `--datasets CAD5,CHI_TAXI` to evaluate a target subset. Both entry points
support `--device cpu`. Set `PYTHON_BIN` to select the Python
interpreter used by the shell scripts.

## Configuration

`configs/topost.json` specifies the model, optimization, partitioning, and datasets.

## Code

- `topost/graph.py`: METIS units, sixteen SSE statistics, and graph packing.
- `topost/model.py`: GIN topology encoder, calendar embeddings, time-conditioned
  cross-attention, generated Q/K/graph projections, spatial propagation, and prediction head.
- `topost/data.py`: benchmark periods, calendar indices, chronological splits, and temporal batches.
- `topost/engine.py`: optimization, checkpoint saving, and evaluation metrics.
- `train.py` and `evaluate.py`: executable training and evaluation entry points.
