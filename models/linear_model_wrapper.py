import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_model import BaseModel  # assuming BaseModel provides common loss function etc.

###############################################################################
# 1. The Custom Linear Regression Model
###############################################################################
class CustomLinearRegressionModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 input_chunk_length: int,
                 use_polynomial_features: bool = False,
                 poly_degree: int = 1,
                 dropout: float = 0.0):
        """
        A linear regression model that accepts a sliding-window input and returns
        a sequence of predictions. The model simply flattens the input (with optional
        polynomial expansion) and applies a single linear transformation.
        
        Parameters:
          - input_dim: number of features per time step.
          - output_dim: number of output values per time step (often 1).
          - input_chunk_length: number of time steps per input window.
          - use_polynomial_features: if True, each feature will be expanded to
            [x, x^2, ..., x^(poly_degree)].
          - poly_degree: the degree of the polynomial expansion (if enabled).
          - dropout: optional dropout applied before the linear transformation.
        """
        super(CustomLinearRegressionModel, self).__init__()
        self.input_chunk_length = input_chunk_length
        self.use_polynomial_features = use_polynomial_features
        self.poly_degree = poly_degree

        # Compute the flattened input dimension.
        flat_dim = input_chunk_length * input_dim
        if use_polynomial_features and poly_degree > 1:
            # Each original feature will be expanded to poly_degree features.
            flat_dim = flat_dim * poly_degree

        # Single linear layer for linear regression.
        self.linear = nn.Linear(flat_dim, input_chunk_length * output_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """
        Expects x of shape (batch_size, input_chunk_length, input_dim) and returns
        predictions of shape (batch_size, input_chunk_length, output_dim).
        """
        batch_size = x.size(0)
        # Flatten the time and feature dimensions.
        x = x.reshape(batch_size, -1)  # shape: (batch_size, input_chunk_length * input_dim)

        # Optionally apply polynomial expansion.
        if self.use_polynomial_features and self.poly_degree > 1:
            # Create a list of tensors: [x^1, x^2, ..., x^(poly_degree)]
            poly_features = [x ** (i + 1) for i in range(self.poly_degree)]
            # Concatenate along the feature dimension.
            x = torch.cat(poly_features, dim=1)

        if self.dropout is not None:
            x = self.dropout(x)

        # Apply the linear transformation.
        out = self.linear(x)
        # Reshape to (batch_size, input_chunk_length, output_dim)
        out = out.reshape(batch_size, self.input_chunk_length, -1)
        return out

###############################################################################
# 2. The Linear Regression Model Wrapper
###############################################################################
class LinearRegressionModelWrapper(BaseModel):
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
        Initialize the linear regression model and its optimizer.

        Required hyperparameters for this model include:
          - "input_chunk_length": the sliding-window length.
          - "reg_type": one of ["none", "l1", "l2"] for regularization.
          - "reg_lambda": the regularization coefficient.
          - "use_polynomial_features": bool flag for polynomial expansion.
          - "poly_degree": degree for the polynomial expansion.
          - "dropout": dropout rate (optional).
          - "lr": learning rate.
          - "batch_size", "n_epochs", "output_dim", "input_dim".
        """
        required_keys = [
            "lr", "batch_size", "n_epochs", "output_dim", "input_dim",
            "input_chunk_length", "reg_type", "reg_lambda", "dropout",
            "use_polynomial_features", "poly_degree"
        ]
        for k in required_keys:
            if k not in params:
                raise ValueError(f"Missing required hyperparameter: {k}")

        # Build the linear regression network.
        self.model = CustomLinearRegressionModel(
            input_dim=params["input_dim"],
            output_dim=params["output_dim"],
            input_chunk_length=params["input_chunk_length"],
            use_polynomial_features=params["use_polynomial_features"],
            poly_degree=params["poly_degree"],
            dropout=params["dropout"]
        ).to(self.device)

        # Set up loss function and optimizer.
        self.loss_fn = params.get("loss_fn_callable", nn.MSELoss())
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.batch_size = params["batch_size"]
        self.n_epochs = params["n_epochs"]
        self.params = params

        if "random_state" in params:
            torch.manual_seed(params["random_state"])
            np.random.seed(params["random_state"])

    def suggest_hyperparameters(self, trial):
        """
        Suggest only the linear model–specific hyperparameters.
        (Common hyperparameters such as input_chunk_length, input_dim, etc.,
         are handled by BaseModel.)
        """
        use_poly = trial.suggest_categorical("use_polynomial_features", [False, True])
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_int("batch_size", 16, 128),
            "n_epochs": trial.suggest_int("n_epochs", 10, 200),
            "reg_type": trial.suggest_categorical("reg_type", ["none", "l1", "l2"]),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True),
            "use_polynomial_features": use_poly,
            "poly_degree": trial.suggest_int("poly_degree", 1, 5) if use_poly else 1,
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "output_dim": 1,  # For univariate output
        }

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the linear regression model on training data using a sliding-window
        approach to create overlapping sequences.
        """
        print(f'fit: x shape: {x.shape}, y shape: {y.shape}')
        self.model.train()

        seq_length = self.params['input_chunk_length']

        # Create overlapping windows if x is 2D.
        if x.ndim == 2:
            num_samples, _ = x.shape
            num_windows = num_samples - seq_length + 1
            x = np.array([x[i:i+seq_length] for i in range(num_windows)])
            # For y, if 1D, reshape each window to (seq_length, 1).
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
                if self.params.get("reg_type", "none") != "none" and self.params.get("reg_lambda", 0.0) > 0:
                    reg_loss = 0.0
                    for param in self.model.parameters():
                        if param.requires_grad:
                            if self.params["reg_type"] == "l1":
                                reg_loss += torch.sum(torch.abs(param))
                            elif self.params["reg_type"] == "l2":
                                reg_loss += torch.sum(param ** 2)
                    loss = loss + self.params["reg_lambda"] * reg_loss

                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(dataset)
            # Optionally, print/log the epoch loss.
            # print(f"Epoch {epoch+1}/{self.n_epochs} Loss: {epoch_loss:.4f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict using the trained linear regression model. Uses the same sliding-window
        approach as in fit.
        """
        self.model.eval()
        seq_length = self.params['input_chunk_length']
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

    def predict_next_step(self, x: np.ndarray) -> float:
        """
        Predict the next step (for walk-forward forecasting) using the trained
        linear regression model.
        """
        self.model.eval()
        seq_length = self.params['input_chunk_length']

        if len(x.shape) != 2:
            raise ValueError("Input data must have shape (num_samples, input_dim).")

        num_samples, _ = x.shape
        if num_samples < seq_length:
            raise ValueError(f"Input data must have at least {seq_length} samples.")

        # Use the last 'seq_length' samples as the window.
        last_chunk = x[-seq_length:]
        last_chunk = torch.tensor(last_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(last_chunk)
        # Return the final time-step's prediction.
        next_value = preds.squeeze(0)[-1, 0].item()
        return next_value
