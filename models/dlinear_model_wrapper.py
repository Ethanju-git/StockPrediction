import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_model import BaseModel  # assuming BaseModel sets common hyperparameters such as input_dim and input_chunk_length

###############################################################################
# 1. The Custom DLinear Model
###############################################################################
class CustomDLinearModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 input_chunk_length: int,
                 mode: str = 'individual',
                 dropout: float = 0.0):
        """
        DLinear model for time series forecasting.

        This model applies a channel-wise (i.e. feature-wise) linear mapping to the 
        input window. It supports two modes:
            - 'individual': each channel has its own linear mapping.
            - 'shared': all channels share the same linear mapping.

        The model assumes that the forecast horizon equals the input_chunk_length.

        Parameters:
            - input_dim: number of features (channels) per time step.
            - input_chunk_length: length of the input time window (and forecast horizon).
            - mode: either 'individual' or 'shared'.
            - dropout: dropout rate applied to the input.
        """
        super(CustomDLinearModel, self).__init__()
        self.input_dim = input_dim
        self.input_chunk_length = input_chunk_length
        self.mode = mode
        if mode not in ['individual', 'shared']:
            raise ValueError("mode must be either 'individual' or 'shared'")
        if mode == 'individual':
            # For each channel d, create a weight matrix of shape (input_chunk_length, input_chunk_length)
            # and a bias vector of shape (input_chunk_length).
            # The weight tensor is of shape (input_dim, input_chunk_length, input_chunk_length)
            self.weight = nn.Parameter(torch.randn(input_dim, input_chunk_length, input_chunk_length))
            self.bias = nn.Parameter(torch.zeros(input_dim, input_chunk_length))
        else:
            # Shared mode: a single weight matrix (input_chunk_length x input_chunk_length) is used for all channels.
            self.weight = nn.Parameter(torch.randn(input_chunk_length, input_chunk_length))
            self.bias = nn.Parameter(torch.zeros(input_chunk_length))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        Forward pass.

        Expects x of shape (batch_size, input_chunk_length, input_dim) and returns
        predictions of the same shape.
        """
        if self.dropout is not None:
            x = self.dropout(x)
            
        if self.mode == 'individual':
            # Apply a separate linear mapping to each channel.
            # For each channel d: out[:, :, d] = x[:, :, d] @ weight[d] + bias[d]
            # Using Einstein summation: output shape becomes (batch_size, input_dim, input_chunk_length)
            out = torch.einsum('bld,dlh->bdh', x, self.weight)
            out = out + self.bias.unsqueeze(0)  # broadcast the bias
            # Permute to obtain shape (batch_size, input_chunk_length, input_dim)
            out = out.transpose(1, 2)
        else:
            # Shared mode: apply the same linear mapping for all channels.
            B, L, D = x.shape
            # Reshape x to (B*D, L)
            x_reshaped = x.transpose(1, 2).reshape(B * D, L)
            # Apply linear mapping: (B*D, L) @ (L, L) then add bias of shape (L)
            out = torch.matmul(x_reshaped, self.weight) + self.bias
            # Reshape back to (B, D, L) and then transpose to (B, L, D)
            out = out.reshape(B, D, L).transpose(1, 2)
        return out

###############################################################################
# 2. The DLinear Model Wrapper (without input_dim and input_chunk_length hyperparameters)
###############################################################################
class DLinearModelWrapper(BaseModel):
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
        Initialize the DLinear model and its optimizer.
        
        Required hyperparameters for this model include:
          - "dlinear_mode": one of ["individual", "shared"].
          - "dropout": dropout rate.
          - "lr": learning rate.
          - "batch_size": batch size.
          - "n_epochs": number of training epochs.
          - "reg_type": one of ["none", "l1", "l2", "elasticnet"].
          - "reg_lambda": regularization coefficient.
          - If "reg_type" is "elasticnet", then "l1_ratio" is required.
        
        Note: The "input_dim" and "input_chunk_length" are assumed to be set by the BaseModel.
        """
        required_keys = [
            "lr", "batch_size", "n_epochs", "dlinear_mode", "dropout",
            "reg_type", "reg_lambda"
        ]
        if params.get("reg_type") == "elasticnet":
            required_keys.append("l1_ratio")
        for k in required_keys:
            if k not in params:
                raise ValueError(f"Missing required hyperparameter: {k}")

        # Access input_dim and input_chunk_length from self.params (provided by BaseModel)
        input_dim = self.params["input_dim"]
        input_chunk_length = self.params["input_chunk_length"]

        self.model = CustomDLinearModel(
            input_dim=input_dim,
            input_chunk_length=input_chunk_length,
            mode=params["dlinear_mode"],
            dropout=params["dropout"]
        ).to(self.device)

        self.loss_fn = params.get("loss_fn_callable", nn.MSELoss())
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.batch_size = params["batch_size"]
        self.n_epochs = params["n_epochs"]
        self.params.update(params)

        if "random_state" in params:
            torch.manual_seed(params["random_state"])
            np.random.seed(params["random_state"])

    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters for the DLinear model.
        
        Note: "input_dim" and "input_chunk_length" are not suggested here because they are provided by BaseModel.
        """
        dlinear_mode = trial.suggest_categorical("dlinear_mode", ["individual", "shared"])
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "n_epochs": trial.suggest_int("n_epochs", 10, 200),
            "dlinear_mode": dlinear_mode,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "reg_type": trial.suggest_categorical("reg_type", ["none", "l1", "l2", "elasticnet"]),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True)
        }
        if params["reg_type"] == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        return params

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the DLinear model on training data using a sliding-window approach to create overlapping sequences.
        """
        print(f'fit: x shape: {x.shape}, y shape: {y.shape}')
        self.model.train()
        seq_length = self.params["input_chunk_length"]

        # Create overlapping windows if x is 2D.
        if x.ndim == 2:
            num_samples, _ = x.shape
            num_windows = num_samples - seq_length + 1
            x = np.array([x[i:i+seq_length] for i in range(num_windows)])
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

                # Optionally add a regularization penalty.
                reg_type = self.params.get("reg_type", "none")
                reg_lambda = self.params.get("reg_lambda", 0.0)
                if reg_type != "none" and reg_lambda > 0:
                    reg_loss = 0.0
                    for param in self.model.parameters():
                        if param.requires_grad:
                            if reg_type == "l1":
                                reg_loss += torch.sum(torch.abs(param))
                            elif reg_type == "l2":
                                reg_loss += torch.sum(param ** 2)
                            elif reg_type == "elasticnet":
                                l1_ratio = self.params.get("l1_ratio", 0.5)
                                l1_loss = torch.sum(torch.abs(param))
                                l2_loss = torch.sum(param ** 2)
                                reg_loss += l1_ratio * l1_loss + (1 - l1_ratio) * l2_loss
                    loss = loss + reg_lambda * reg_loss

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)
            # Optionally, log the epoch loss.
            # print(f"Epoch {epoch+1}/{self.n_epochs} Loss: {epoch_loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the trained DLinear model with the same sliding-window approach.
        """
        self.model.eval()
        seq_length = self.params["input_chunk_length"]
        print(f'predict: x shape: {x.shape}')

        if torch.is_tensor(x):
            x = x.cpu().numpy()
        if x.ndim == 2:
            num_samples, _ = x.shape
            num_windows = num_samples - seq_length + 1
            x = np.array([x[i:i+seq_length] for i in range(num_windows)])
        with torch.no_grad():
            X_test_t = torch.tensor(x, dtype=torch.float32).to(self.device)
            preds = self.model(X_test_t)
            return preds.cpu().numpy()

    def predict_next_step(self, x: np.ndarray):
        """
        Predict the next step (for walk-forward forecasting) using the trained DLinear model.
        
        Returns a scalar for the target ticker by selecting the first channel of the last time-step.
        """
        self.model.eval()
        seq_length = self.params["input_chunk_length"]
        if len(x.shape) != 2:
            raise ValueError("Input data must have shape (num_samples, input_dim).")
        num_samples, _ = x.shape
        if num_samples < seq_length:
            raise ValueError(f"Input data must have at least {seq_length} samples.")

        last_chunk = x[-seq_length:]
        last_chunk = torch.tensor(last_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(last_chunk)
        # preds has shape (1, seq_length, input_dim); select the last time-step and the first channel.
        next_value = preds.squeeze(0)[-1, 0].item()
        return next_value
