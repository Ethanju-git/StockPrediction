import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_model import BaseModel
import xgboost as xgb



# from .base_model import BaseModel  # Uncomment/modify based on actual project structure


class XGBModelWrapper(BaseModel):
    """
    A model wrapper that uses XGBoost for regression and replicates the 
    rolling window training approach used in the RNN code.
    """
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
        """
        Initialize the model wrapper. Notice that `input_chunk_length` is not
        set here; it will be determined during hyperparameter tuning (similar
        to the RNN code).
        """
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
            max_num_features = max_num_features
        )

        self.input_chunk_length = None  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # This will store the actual XGBoost model after initialization
        self.xgb_model = None

        # Store hyperparameters here after they are set
        self.params = {}

    def initialize_model(self, params: dict):
        """
        Initialize the XGBoost model from a dictionary of parameters.
        One of these parameters is `input_chunk_length`.
        """
        required_keys = [
            "input_chunk_length", "n_estimators", "max_depth", "learning_rate", 
            "subsample", "colsample_bytree", "random_state", "tree_method"
        ]
        for k in required_keys:
            if k not in params:
                raise ValueError(f"Missing required hyperparameter: {k}")

        # Now set the input_chunk_length from the chosen hyperparameters
        self.input_chunk_length = params["input_chunk_length"]
        self.params = params

        # You can tune/adjust other XGBoost parameters as needed
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            random_state=params["random_state"],
            tree_method=params["tree_method"],  # e.g., "gpu_hist" for GPU training
        )

    def suggest_hyperparameters(self, trial):
        """
        Suggest hyperparameters using an Optuna trial.
        This example includes 'input_chunk_length' to mirror the RNN approach.
        Adjust ranges based on your domain knowledge.
        """
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
            # If you want to auto-detect GPU usage, you could do so here
            # For now, let's just default to "gpu_hist" if GPU is available, else "auto".
            "tree_method": "gpu_hist" if torch.cuda.is_available() else "auto",
        }

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit the XGBoost model on training data in a many-to-1 style using 
        a rolling window approach to match the logic of the RNN code.
        """
        # Ensure y is at least 2D (though for XGBoost, 1D is typically fine; we keep the same shape for consistency).
        if y.ndim == 2 and y.shape[1] == 1:
            # Flatten it to shape (num_samples,)
            y = y.squeeze(axis=1)
        elif y.ndim > 2:
            raise ValueError("y cannot have more than 2 dimensions for XGBoost regression.")

        seq_length = self.input_chunk_length
        num_samples = x.shape[0]
        input_dim = x.shape[1]

        # Build rolling windows using list comprehensions.
        X_list = [x[i : i + seq_length].reshape(-1) for i in range(num_samples - seq_length + 1)]
        Y_list = [y[i + seq_length - 1] for i in range(num_samples - seq_length + 1)]

        X_rolled = np.array(X_list)  # shape: (num_sequences, seq_length * input_dim)
        y_rolled = np.array(Y_list)  # shape: (num_sequences,)

        # Train the XGBoost model
        self.xgb_model.fit(X_rolled, y_rolled)


    def predict_next_step(self, x: np.ndarray) -> float:
        """
        Predict the next step (a single scalar) using the trained XGBoost model.
        `x` must have at least `self.input_chunk_length` rows (time steps).

        We flatten the last chunk and feed it to XGBoost.
        """
        if len(x.shape) != 2:
            raise ValueError("Input data must have shape (num_samples, input_dim).")
        num_samples, input_dim = x.shape
        if num_samples < self.input_chunk_length:
            raise ValueError(
                f"Input data must have at least {self.input_chunk_length} samples. Got {num_samples}."
            )

        # Extract the last chunk
        last_chunk = x[-self.input_chunk_length:]  # (seq_length, input_dim)

        # Flatten the chunk for XGBoost
        last_chunk_flat = last_chunk.reshape(1, -1)  # shape: (1, seq_length*input_dim)

        pred = self.xgb_model.predict(last_chunk_flat)
        # pred is a 1D array with a single value
        return float(pred[0])


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions for each valid window in `x`. For a sequence of length N,
        this will output (N - input_chunk_length + 1) predictions.
        """
        total_preds = np.array([
            self.predict_next_step(x[i : i + self.input_chunk_length])
            for i in range(len(x) - self.input_chunk_length + 1)
        ])
        return total_preds
