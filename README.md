# Gesture Recognition with Recurrent Neural Networks

Classifying arm gestures from smartwatch accelerometer data using PyTorch RNNs. Built as part of a machine learning course, this project experiments with GRU, LSTM, and Conv1d+GRU architectures and uses Hyperopt to automate hyperparameter search.

## Dataset

The [SmartWatch Gestures Dataset](https://tev.fbk.eu/resources/smartwatch) contains 3200 sequences collected from 8 users performing 20 different gestures, 20 repetitions each. Each sequence is a variable-length time series of 3-axis accelerometer readings (x, y, z) from a Sony SmartWatch worn on the right wrist.

| Property | Value |
|---|---|
| Classes | 20 gestures |
| Sequences | 3200 total |
| Features | 3 (accelerometer x, y, z) |
| Sequence length | Variable (padded per batch) |
| Blind guess accuracy | 5% |

## Project structure

```
├── gestures_exercise.py       # All models with explanations — start here
├── gestures_hyperopt.py       # Automated hyperparameter search with Hyperopt
├── mlflow_database.db         # MLflow experiment tracking (auto-generated)
├── models/                    # Saved model checkpoints (auto-generated)
├── gestures/                  # Training logs (auto-generated)
└── README.md
```

## Models

Three architectures are implemented, each building on the previous:

**GRUModel** — baseline. Feeds raw accelerometer readings directly into a GRU, takes the last hidden state, and classifies with a linear layer.

**LSTMModel** — same structure as GRUModel but uses an LSTM cell, which adds a separate cell state for longer-range memory.

**Conv1dGRU** — best performing. A Conv1d layer first slides along the time axis to extract local patterns (peaks, direction changes), then passes the richer feature sequence to a GRU. This is the architecture used in the hyperparameter search.

```
Input (batch, T, 3)
    │
    ├── GRUModel:    GRU → last hidden state → Linear → 20 logits
    ├── LSTMModel:   LSTM → last hidden state → Linear → 20 logits
    └── Conv1dGRU:   Conv1d → GRU → last hidden state → Linear → 20 logits
```

## Installation

```bash
git clone https://github.com/IqraRafiq/gesture-recognition.git
cd gesture-recognition

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install torch mads-datasets mltrainer mlflow hyperopt loguru
```

## Running the exercises

### 1. Train all models

Run `gestures_exercise.py` to train and compare all architectures. Each model is logged to MLflow automatically.

```bash
python gestures_exercise.py
```

Or run it cell by cell in a Jupyter notebook.

### 2. Hyperparameter search

Run `gestures_hyperopt.py` to let Hyperopt search for the best combination of `hidden_size`, `num_layers`, `conv_filters`, and `kernel_size` across multiple training runs.

```bash
python gestures_hyperopt.py
```

Set `max_evals` in `main()` to control how many trials to run. Start with 3 to verify it works, then increase to 20–50 for meaningful results.

### 3. View results in MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_database.db
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. You can compare runs, plot accuracy curves per epoch, and filter by model or hyperparameter value.

> **Windows users:** do not use the `file://` URI prefix when setting the tracking URI — pass the path string directly or use the `sqlite:///` format shown above.

## Hyperparameters searched

| Parameter | Range | Description |
|---|---|---|
| `hidden_size` | 32 – 256 | GRU hidden state size |
| `num_layers` | 1 – 3 | Number of stacked GRU layers |
| `conv_filters` | 16 – 64 | Conv1d output channels |
| `kernel_size` | 3 – 7 | Conv1d temporal window size |

## Requirements

- Python 3.9+
- PyTorch
- mads-datasets
- mltrainer
- mlflow
- hyperopt
- loguru
