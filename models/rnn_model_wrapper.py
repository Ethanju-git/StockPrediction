import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_model import BaseModel


class CustomRNN(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 rnn_type: str, 
                 n_rnn_layers: int, 
                 dropout: float, 
                 output_dim: int):
        super(CustomRNN, self).__init__()
        
        # Validate the RNN type
        if rnn_type not in ["RNN", "LSTM", "GRU"]:
            raise ValueError("rnn_type must be one of ['RNN', 'LSTM', 'GRU'].")

        self.rnn_type = rnn_type
        self.hidden_dim = hidden_dim
        self.n_rnn_layers = n_rnn_layers

        # Define the RNN layer
        if rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=input_dim, 
                hidden_size=hidden_dim, 
                num_layers=n_rnn_layers, 
                batch_first=True, 
                dropout=dropout if n_rnn_layers > 1 else 0
            )
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim, 
                hidden_size=hidden_dim, 
                num_layers=n_rnn_layers, 
                batch_first=True, 
                dropout=dropout if n_rnn_layers > 1 else 0
            )
        else:  # GRU
            self.rnn = nn.GRU(
                input_size=input_dim, 
                hidden_size=hidden_dim, 
                num_layers=n_rnn_layers, 
                batch_first=True, 
                dropout=dropout if n_rnn_layers > 1 else 0
            )

        # Final linear layer to map RNN outputs to a single target output
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length, input_dim)
        if len(x.shape) != 3:
            raise ValueError("Input tensor must have shape (batch_size, seq_length, input_dim)")

        if self.rnn_type == "LSTM":
            out, (h, c) = self.rnn(x, hidden)
        else:
            out, h = self.rnn(x, hidden)

        # Ensure output has at least 3 dimensions
        if out.dim() != 3:
            raise ValueError("RNN output does not have 3 dimensions as expected.")

        out = self.fc(out)  # (batch_size, seq_length, output_dim)
        return out


class RNNModelWrapper(BaseModel):
    def __init__(
        self,
        target_ticker,
        other_tickers,
        start_date,
        end_date,
        init_train_perc,
        min_steps_per_retune,
        is_regressive,
        max_n_correlated_stocks,
        num_trials_per_tuning,
        max_num_features
    ):
        super().__init__(
            target_ticker=target_ticker,
            other_tickers=other_tickers,
            start_date=start_date,
            end_date=end_date,
            init_train_perc=init_train_perc,
            min_steps_per_retune=min_steps_per_retune,
            is_regressive=is_regressive,
            max_n_correlated_stocks=max_n_correlated_stocks,
            num_trials_per_tuning=num_trials_per_tuning,
            max_num_features=max_num_features
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.params = {}

    def initialize_model(self, params: dict):
        """
        Initialize the custom RNN model and its optimizer.
        This file now does not perform any loss function selection; it uses the common loss
        function callable provided (via params or set by BaseModel).
        """
        # Make sure RNN-specific hyperparameters are present
        required_keys = [
            "hidden_dim", "rnn_type", "n_rnn_layers", "dropout",
            "lr", "batch_size", "n_epochs", "output_dim", "input_dim"
        ]
        for k in required_keys:
            if k not in params:
                raise ValueError(f"Missing required hyperparameter: {k}")

        # Build the actual RNN
        self.model = CustomRNN(
            input_dim=params["input_dim"],       # <-- CHANGED: read from params
            hidden_dim=params["hidden_dim"],
            rnn_type=params["rnn_type"],
            n_rnn_layers=params["n_rnn_layers"],
            dropout=params["dropout"],
            output_dim=params["output_dim"]
        ).to(self.device)

        # Use the loss function from params (BaseModel sets that up)
        self.loss_fn = params.get("loss_fn_callable", torch.nn.MSELoss())

        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.batch_size = params["batch_size"]
        self.n_epochs = params["n_epochs"]
        self.params = params

        if "random_state" in params:
            torch.manual_seed(params["random_state"])
            np.random.seed(params["random_state"])

    def suggest_hyperparameters(self, trial):
        """
        Suggest only RNN-specific hyperparameters.
        Common hyperparameters (loss_func, input_chunk_length, etc.)
        are handled by BaseModel.
        """
        return {
            "hidden_dim": trial.suggest_int("hidden_dim", 10, 200),
            "rnn_type": trial.suggest_categorical("rnn_type", ["RNN", "LSTM", "GRU"]),
            "n_rnn_layers": trial.suggest_int("n_rnn_layers", 1, 10),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "n_epochs": trial.suggest_int("n_epochs", 10, 200),
            "output_dim": 1,  # For univariate output
        }

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the RNN model on training data.
        Now uses a manual sliding window approach to create overlapping sequences,
        e.g., [x0, x1], [x1, x2], [x2, x3], ...
        """
        self.model.train()
        
        seq_length = self.params['input_chunk_length']  # <-- CHANGED

        # Create overlapping windows if x is 2D
        if x.ndim == 2:
            num_samples, input_dim = x.shape
            num_windows = num_samples - seq_length + 1
            x = np.array([x[i:i+seq_length] for i in range(num_windows)])
            # For y, if it's a 1D array, reshape each window to (seq_length, 1)
            if y.ndim == 1:
                y = np.array([y[i:i+seq_length][:, None] for i in range(num_windows)])
            else:
                y = np.array([y[i:i+seq_length] for i in range(num_windows)])

        X_train_t = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)
            # Optionally print or log: print(f"Epoch {epoch+1}/{self.n_epochs} Loss: {epoch_loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the trained RNN model.
        Now uses a manual sliding window approach to create overlapping sequences for prediction.
        """
        self.model.eval()

        seq_length = self.params['input_chunk_length']  # <-- CHANGED

        if torch.is_tensor(x):
            x = x.cpu().numpy()

        # Create overlapping windows if x is 2D
        if x.ndim == 2:
            num_samples, input_dim = x.shape
            num_windows = num_samples - seq_length + 1
            x = np.array([x[i:i+seq_length] for i in range(num_windows)])

        with torch.no_grad():
            X_test_t = torch.tensor(x, dtype=torch.float32).to(self.device)
            preds = self.model(X_test_t)
            return preds.cpu().numpy()

    def predict_next_step(self, x: np.ndarray) -> float:
        """
        Predict the next step using the trained RNN model (e.g., walk-forward).
        """
        self.model.eval()

        seq_length = self.params['input_chunk_length']  # <-- CHANGED
        if len(x.shape) != 2:
            raise ValueError("Input data must have shape (num_samples, input_dim).")

        num_samples, input_dim = x.shape
        if num_samples < seq_length:
            raise ValueError(f"Input data must have at least {seq_length} samples.")

        # Take the last 'seq_length' samples
        last_chunk = x[-seq_length:]
        last_chunk = torch.tensor(last_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(last_chunk)
        # Return the final time step's prediction in that sequence
        next_value = preds.squeeze(0)[-1, 0].item()
        return next_value
