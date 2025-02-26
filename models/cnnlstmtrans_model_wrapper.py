import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_model import BaseModel

class LearnedCyclicPositionalEncoding(nn.Module):
    """
    Learned cyclic positional encoding that combines:
      1. A learned global (absolute) positional embedding (one per time step up to max_len)
      2. A learned cyclic embedding for the business week (5 days)
      3. A learned cyclic embedding for the business month (25 days)
      4. A learned cyclic embedding for the business year (252 days)
      
    The overall embedding dimension (d_model) is split into four parts:
      - d_model_global for the global embedding,
      - and the remaining dimension is equally split among week, month, and year.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000, week=5, month=25, year=252):
        super(LearnedCyclicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Split d_model: reserve a chunk for global learned positions and the rest for cyclic components.
        d_model_part = d_model // 4  
        d_model_global = d_model - 3 * d_model_part  # ensures the total sums to d_model
        
        self.d_model = d_model
        
        # Global learned positional embeddings: one for each position up to max_len.
        self.global_pe = nn.Embedding(max_len, d_model_global)
        
        # Learned cyclic embeddings:
        self.week_pe = nn.Embedding(week, d_model_part)
        self.month_pe = nn.Embedding(month, d_model_part)
        self.year_pe = nn.Embedding(year, d_model_part)
        
        # Initialize weights
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.uniform_(self.global_pe.weight, -0.1, 0.1)
        nn.init.uniform_(self.week_pe.weight, -0.1, 0.1)
        nn.init.uniform_(self.month_pe.weight, -0.1, 0.1)
        nn.init.uniform_(self.year_pe.weight, -0.1, 0.1)
        
    def forward(self, x):
        """
        x shape: (batch_size, seq_length, d_model)
        """
        batch_size, seq_len, _ = x.size()
        
        # Global learned positional embeddings:
        pos_global = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        global_emb = self.global_pe(pos_global)  # shape: (batch_size, seq_len, d_model_global)
        
        # Learned cyclic embeddings:
        pos = torch.arange(seq_len, device=x.device)  # shape: (seq_len,)
        week_indices = (pos % self.week_pe.num_embeddings).unsqueeze(0).expand(batch_size, seq_len)
        week_emb = self.week_pe(week_indices)  # shape: (batch_size, seq_len, d_model_part)
        
        month_indices = (pos % self.month_pe.num_embeddings).unsqueeze(0).expand(batch_size, seq_len)
        month_emb = self.month_pe(month_indices)  # shape: (batch_size, seq_len, d_model_part)
        
        year_indices = (pos % self.year_pe.num_embeddings).unsqueeze(0).expand(batch_size, seq_len)
        year_emb = self.year_pe(year_indices)  # shape: (batch_size, seq_len, d_model_part)
        
        # Concatenate embeddings along the last dimension to obtain a tensor of shape (batch_size, seq_len, d_model)
        pe_total = torch.cat([global_emb, week_emb, month_emb, year_emb], dim=-1)
        
        x = x + pe_total
        return self.dropout(x)

class CNNLSTMTransformer(nn.Module):
    """
    A hybrid architecture that:
      1) Applies a 1D convolution over time,
      2) Feeds the output to an LSTM,
      3) Passes the LSTM output through a Transformer encoder,
      4) Outputs a single (many-to-1) prediction from the final time step of the Transformer.
    """
    def __init__(
        self,
        input_dim: int,
        n_filters: int,
        kernel_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        d_model: int,
        nhead: int,
        num_transformer_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_dim: int = 1,
    ):
        super(CNNLSTMTransformer, self).__init__()

        # ----- CNN Stage -----
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=n_filters,
            kernel_size=kernel_size
        )

        # ----- LSTM Stage -----
        self.lstm = nn.LSTM(
            input_size=n_filters, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        # Project LSTM output to d_model (if needed) before passing it to the Transformer.
        self.projection = (
            nn.Linear(lstm_hidden_size, d_model) if (lstm_hidden_size != d_model) else nn.Identity()
        )

        # ----- Learned Cyclic Positional Encoding -----
        self.pos_encoder = LearnedCyclicPositionalEncoding(d_model, dropout=dropout)

        # ----- Transformer Encoder Stage -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # This ensures the input shape remains (batch_size, seq_length, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # ----- Final Projection -----
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Forward pass (many-to-1):
          - x shape: (batch_size, seq_length, input_dim)

        Returns:
          - preds shape: (batch_size, output_dim)
        """
        # 1) CNN expects (batch_size, input_dim, seq_length)
        x = x.permute(0, 2, 1)  # (N, C_in, L_in)
        conv_out = self.conv1d(x)  
        # conv_out shape: (batch_size, n_filters, L_out) 
        # where L_out = seq_length - kernel_size + 1 (if stride=1 and no padding)

        # 2) Permute for LSTM: (batch_size, L_out, n_filters)
        conv_out = conv_out.permute(0, 2, 1)

        # 3) LSTM stage
        lstm_out, (h_n, c_n) = self.lstm(conv_out)
        # lstm_out shape: (batch_size, L_out, lstm_hidden_size)

        # 4) Project LSTM output to d_model if necessary
        x_proj = self.projection(lstm_out)
        # shape: (batch_size, L_out, d_model)

        # 5) Add learned cyclic positional encoding and pass through Transformer encoder
        x_pe = self.pos_encoder(x_proj)  # (batch_size, L_out, d_model)
        transformer_out = self.transformer_encoder(x_pe)
        # shape: (batch_size, L_out, d_model)

        # 6) Many-to-1: Take the last time step from the Transformer output
        final_out = transformer_out[:, -1, :]  # shape: (batch_size, d_model)

        # 7) Final linear projection to output dimension
        preds = self.fc_out(final_out)  # shape: (batch_size, output_dim)

        return preds

class CNNLSTMTransformerModelWrapper(BaseModel):
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
        self.input_chunk_length = None

    def initialize_model(self, params: dict):
        """
        Initialize the CNN-LSTM-Transformer model with a dict of hyperparameters.
        """
        required_keys = [
            "input_chunk_length",
            "n_filters",
            "kernel_size",
            "lstm_hidden_size",
            "lstm_num_layers",
            "d_model",
            "nhead",
            "num_transformer_layers",
            "dim_feedforward",
            "dropout",
            "lr",
            "batch_size",
            "n_epochs",
            "output_dim",
            "input_dim",
        ]
        for k in required_keys:
            if k not in params:
                raise ValueError(f"Missing hyperparam: {k}")

        self.input_chunk_length = params["input_chunk_length"]
        self.params = params

        # Build the model
        self.model = CNNLSTMTransformer(
            input_dim=params["input_dim"],
            n_filters=params["n_filters"],
            kernel_size=params["kernel_size"],
            lstm_hidden_size=params["lstm_hidden_size"],
            lstm_num_layers=params["lstm_num_layers"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_transformer_layers=params["num_transformer_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params["dropout"],
            output_dim=params["output_dim"],
        ).to(self.device)

        # Set up optimizer, loss function, etc.
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.batch_size = params["batch_size"]
        self.n_epochs = params["n_epochs"]

        # Optionally set random seeds
        if "random_state" in params:
            torch.manual_seed(params["random_state"])
            np.random.seed(params["random_state"])

    def suggest_hyperparameters(self, trial):
        # 1) Choose nhead from 2..8 in increments of 2, for example
        nhead = trial.suggest_int("nhead", 2, 8, step=2)
        
        d_model = trial.suggest_int("d_model", 16, 512)
        
        if d_model % nhead != 0:
            temp = d_model / nhead
            if (temp - int(temp)) > 0.5:
                while d_model % nhead != 0:
                    d_model = d_model + 1 
            else:
                while d_model % nhead != 0:
                    d_model = d_model - 1

        input_chunk_length = trial.suggest_int("input_chunk_length", 10, 100)
        return {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 10, 100),
            "n_filters": trial.suggest_int("n_filters", 16, 128, step=16),
            "kernel_size": trial.suggest_int("kernel_size", 2, input_chunk_length),
            "lstm_hidden_size": trial.suggest_int("lstm_hidden_size", 32, 256, step=32),
            "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 10),
            "num_transformer_layers": trial.suggest_int("num_transformer_layers", 1, 10),
            "dim_feedforward": trial.suggest_int("dim_feedforward", 128, 2048, step=128),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128, step=16),
            "n_epochs": trial.suggest_int("n_epochs", 10, 200, step=10),
            "nhead": nhead,
            "d_model": d_model,
            "output_dim": 1,
            "random_state": 42,
        }

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the model using a rolling-window approach (many-to-1).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model(params) first.")

        self.model.train()

        # Check that x has at least the required number of columns (input_dim)
        if x.shape[1] < self.params["input_dim"]:
            raise ValueError(
                f"x has {x.shape[1]} columns, but input_dim is {self.params['input_dim']}."
            )
        # Slice x if more columns than needed
        x_sliced = x[:, :self.params["input_dim"]]

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        seq_length = self.input_chunk_length
        num_samples = x_sliced.shape[0]

        # Create rolling windows
        X_list, Y_list = [], []
        for i in range(num_samples - seq_length + 1):
            X_list.append(x_sliced[i : i + seq_length])  # shape: (seq_length, input_dim)
            Y_list.append(y[i + seq_length - 1])           # shape: (1,)

        X_rolled = np.array(X_list)  # shape: (num_sequences, seq_length, input_dim)
        y_rolled = np.array(Y_list)  # shape: (num_sequences, 1)

        # Convert to torch tensors
        X_train_t = torch.tensor(X_rolled, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_rolled, dtype=torch.float32).to(self.device)

        # DataLoader
        dataset = TensorDataset(X_train_t, y_train_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                preds = self.model(X_batch)  # (batch_size, output_dim)
                loss = self.loss_fn(preds, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(dataset)
            # Optionally, print or log the epoch loss:
            # print(f"Epoch [{epoch+1}/{self.n_epochs}] - Loss: {epoch_loss:.6f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for each valid rolling window in x.
        Output shape: (N - input_chunk_length + 1,)
        """
        self.model.eval()

        # Slice x as before
        if x.shape[1] < self.params["input_dim"]:
            raise ValueError(
                f"x has {x.shape[1]} columns, but input_dim is {self.params['input_dim']}."
            )
        x_sliced = x[:, :self.params["input_dim"]]

        seq_length = self.input_chunk_length
        total_preds = []
        for i in range(len(x_sliced) - seq_length + 1):
            input_series = x_sliced[i : i + seq_length]
            pred = self.predict_next_step(input_series)
            total_preds.append(pred)
        return np.array(total_preds)

    def predict_next_step(self, x: np.ndarray) -> float:
        """
        Predict a single step given at least `input_chunk_length` time steps.
        """
        self.model.eval()

        if x.shape[0] < self.input_chunk_length:
            raise ValueError(
                f"Need at least {self.input_chunk_length} time steps. Got {x.shape[0]}."
            )

        # Slice to correct shape
        x_sliced = x[-self.input_chunk_length:, :self.params["input_dim"]]

        # Add batch dimension and convert to tensor
        x_tensor = torch.tensor(x_sliced, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(x_tensor)  # shape: (1, 1)
        return pred.item()
