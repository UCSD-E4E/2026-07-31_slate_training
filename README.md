# slate-training

Model-assisted **dive-slate labeling** for the FishSense laser-calibration
pipeline — the last human bottleneck before a dive can be calibrated.

Given a rectified underwater frame containing a known calibration slate,
predict the slate's ordered reference points so a `LaserExtrinsics`
calibration can be computed without a human labeler (or with a human only
confirming a pre-annotation).

Delivery mirrors the existing laser detector: a versioned checkpoint consumed
by the data-worker as a GPU `predict` activity that seeds Label Studio
pre-annotations.

## Status

Phase 0 — data + baseline. See [docs/design.md](docs/design.md) for the
label contract and the approach decision.

## Layout

| Path | What |
| --- | --- |
| `src/slate_training/contracts.py` | The `DiveSlateLabel` coordinate/order contract (pure logic) |
| `src/slate_training/extract.py` | Dataset extraction via the fishsense-api SDK |
| `tests/` | `pytest`; `-m integration` needs live credentials |
| `data/` | Vendored label exports (gitignored images) |

## Develop

```sh
uv sync --group dev
uv run pytest -q            # pure-logic tests, no credentials needed
uv run pytest -m integration  # needs API credentials
```
