class BaseModel(ABC):
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
        num_trials_per_tuning
    ):
        self.target_ticker = target_ticker
        self.other_tickers = other_tickers
        self.start_date = start_date
        self.end_date = end_date
        self.init_train_perc = init_train_perc
        self.min_steps_per_retune = min_steps_per_retune
        self.is_regressive = is_regressive
        self.max_n_correlated_stocks = max_n_correlated_stocks
        self.num_trials_per_tuning = num_trials_per_tuning
        self.model = None
        self.unseen_X = None
        self.unseen_y = None
        self.seen_X = None
        self.seen_y = None
        self.time_idx = None
        self.params = {}  # Will store the best hyperparameters after tuning
        self.best_error = None
        self.total_predictions = None
        self.total_actuals = None
        self.log_file = None  # Optionally set this from the outside if desired
        self.steps_since_tuned = 0
        self.features_per_stock = None
        self.scaler_X = None
        self.downloaded_data = None
        self.input_dim = None
        self.warm_start_trials = None

    @abstractmethod
    def initialize_model(self, params):
        """Initialize the model with provided hyperparameters."""
        pass

    @abstractmethod
    def suggest_hyperparameters(self, trial):
        """Suggest model-specific hyperparameters."""
        pass

    @abstractmethod
    def predict(self, x):
        pass
    
    @abstractmethod
    def predict_next_step(self, x):
        pass
    
    @abstractmethod
    def fit(self, x, y):
        pass
    
    def suggest_common_hyperparams(self, trial):
        #
        params = {
            "input_chunk_length": trial.suggest_int("input_chunk_length", 1, 100),
            "num_features": trial.suggest_int("num_features", 1, 100),
            "reduce_dimensionality": trial.suggest_categorical('reduce_dimensionality', [True, False]),
            "heuristic": trial.suggest_categorical('heuristic', ['greedy', 'mrmr']),
            "scaler": trial.suggest_categorical('scaler', ['minmax', 'standard', 'yeo-johnson', 'quantile-normal', 'quantile-uniform']),
            "adversarial": trial.suggest_categorical('adversarial', [True, False]),
            'loss_func': trial.suggest_categorical('loss_func', ['mae', 'mse', 'linex', 'quantile']),
            'weighting_factor': trial.suggest_float('weighting_factor', 0.6, 1) #value of 1 represents no weighting bc 1^n = 1
        }
        
        if params['reduce_dimensionality'] == True:
            max_pca_components = min(params['input_chunk_length'], params['num_features']) - 1
            max_pca_components = min(10, max_pca_components)
            if max_pca_components < 1:
                params['reduce_dimensionality'] = False



        if params['loss_func'] == 'mae':
            params['weigh_under'] = trial.suggest_float('weigh_under', 1, 3)
        elif params['loss_func'] == 'mse':
            params['weigh_under'] = trial.suggest_float('weigh_under', 1, 3)
        elif params['loss_func'] == 'linex':
            params['linex_a'] = trial.suggest_float('linex_a', 0.1, 2.0)
        elif params['loss_func'] == 'quantile':
            params['quantile_tau'] = trial.suggest_float('quantile_tau', 0.1, 0.9)
                
        if params['adversarial'] == True:
            params['discount_factor'] = trial.suggest_float('discount_factor', 0.2, 0.99)
            params['risk'] = trial.suggest_categorical('risk', ['Conservative', 'Liberal'])
            
        if not self.is_regressive:
            params['num_classes'] = trial.suggest_int('num_classes', 2, 10)

        # Dimensionality reduction-related hyperparameters
        if params["reduce_dimensionality"]:
            # Suggest the type of dimensionality reducer, removed the others due to library update making them unusable for now
            params["dimensionality_reducer"] = 'pca'

        return params


    def tune_hyperparameters(self):
        """
        Tunes hyperparameters using Optuna. Logs trial information to a file
        specified by self.log_file. If self.log_file is None, defaults to
        'hyperparameter_tuning.log'.

        Additionally, if warm-start trials from a previous call exist,
        they are injected into the study with their profit values divided by 2.
        This encourages exploration since the data has changed.
        """
        generate_data(self)
        # Decide the log file name (can be set externally, or defaults to .log)
        log_file = self.log_file or "hyperparameter_tuning.log"
        
        def calculate_ema_performance(performance_values, alpha):
            """
            Computes the Exponential Moving Average (EMA) of a given performance metric.

            :param performance_values: List or NumPy array of performance metric values.
            :param alpha: The smoothing factor (between 0 and 1).
            :return: NumPy array of EMA values.
            """
            print(performance_values.shape)
            ema_values = np.zeros_like(performance_values, dtype=np.float64)
            ema_values[0] = performance_values[0]  # Initialize EMA with first value

            for t in range(1, len(performance_values)):
                ema_values[t] = alpha * performance_values[t] + (1 - alpha) * ema_values[t - 1]
            
            return ema_values
        
        def generate_masked_array(mEMA, sEMA, mode="conservative"):
            """
            Generates a masked array based on the provided logic.

            :param mEMA: NumPy array of main EMA values.
            :param sEMA: NumPy array of synthetic EMA values.
            :param mode: Decision mode - "conservative" or "liberal".
            :return: NumPy array with values {-1, 0, 1}.
            """
            assert len(mEMA) == len(sEMA), "Arrays must have the same length."

            masked_array = np.zeros_like(mEMA, dtype=int)

            # Condition 1: Use model (expected regime)
            mask_model = (mEMA > 0) & (sEMA < 0)
            masked_array[mask_model] = 1

            # Condition 2: Use synth (inverse regime detected)
            mask_synth = (mEMA < 0) & (sEMA > 0)
            masked_array[mask_synth] = -1

            # Condition 3: Conflicting signals
            mask_conflict = (mEMA > 0) & (sEMA > 0)
            if mode == "conservative":
                masked_array[mask_conflict] = 0  # Stay out
            else:  # Liberal mode
                masked_array[mask_conflict] = 1  # Use model

            # Condition 4: Both models failing
            mask_fail = (mEMA < 0) & (sEMA < 0)
            if mode == "conservative":
                masked_array[mask_fail] = 0  # Stay out
            else:  # Liberal mode
                masked_array[mask_fail] = -1  # Invert predictions

            return masked_array


        def objective(trial):
            # Suggest the "common" hyperparams
            generic_model_params = self.suggest_common_hyperparams(trial)
            self.params = generic_model_params  # needed for dynamic data construction
            
            # Suggest the model-specific hyperparams
            individual_model_params = self.suggest_hyperparameters(trial)
            model_params = {**generic_model_params, **individual_model_params}
            self.params = model_params
            
            #prepare data and copy it in
            prepare_data(self)
            xseries = self.seen_X
            yseries = self.seen_y

            # Log the start of this trial
            with open(log_file, "a") as f:
                f.write(f"\n=== Starting trial {trial.number} ===\n")
                f.write(f"Parameters: {model_params}\n")

            # Divide the series into 5 sequential blocks (you can adjust n_blocks as needed)
            n_blocks = 5
            total_length = len(yseries)
            block_size = total_length // n_blocks
            indices = np.arange(total_length)
            # We'll do a reverse approach here so the trial pruning starts with more recent data
            indices_reversed = indices[::-1]
            iblocks = [
                indices_reversed[i * block_size : (i + 1) * block_size]
                for i in range(n_blocks)
            ]

            pairs = []
            for i in range(n_blocks):
                # Validation block
                ival_block = iblocks[i]
                # Training is all the other blocks
                itrain_block = np.concatenate(
                    [block for j, block in enumerate(iblocks) if j != i]
                )
                pairs.append((ival_block, itrain_block))

            total_profit = 0.0
            for idx, (ival, itrain) in enumerate(pairs):
                xval = xseries[ival]
                xtrain = xseries[itrain]
                yval = yseries[ival]
                ytrain = yseries[itrain]

                # Initialize and fit the model
                self.initialize_model(model_params)
                self.fit(xtrain, ytrain)

                # Generate forecasts for the validation block, one step at a time
                forecast = []
                for index in ival:
                    start_chunk = index - self.params['input_chunk_length']
                    if start_chunk > 0:
                        input_data = xseries[index - self.params['input_chunk_length']: index]
                        pred = self.predict_next_step(input_data)
                        forecast.append(pred)
                    else:
                        if self.is_regressive:
                            forecast.append(0)
                        else:
                            forecast.append(np.zeros((1, self.params['num_classes'])))  # or torch.zeros((1, self.num_classes))


                # Check for NaNs in forecast
                if torch.is_tensor(forecast):
                    # Detach the tensor from the computation graph and move it to CPU
                    forecast = forecast.detach().cpu().numpy()
                elif isinstance(forecast, list):
                    # If it's a list, convert each element if it's a tensor
                    forecast = np.array([item.detach().cpu().numpy() if torch.is_tensor(item) else item for item in forecast])
                elif isinstance(forecast, dict):
                    # If it's a dict, handle tensor values
                    forecast = {key: (value.detach().cpu().numpy() if torch.is_tensor(value) else value) for key, value in forecast.items()}
                else:
                    # Handle other possible types if necessary
                    forecast = np.array(forecast)
                num_nans = np.isnan(forecast).sum()
                if num_nans > 0:
                    total_elements = len(forecast)
                    nan_percentage = (num_nans / total_elements) * 100
                    raise ValueError(
                        f"Invalid forecast values (NaN or Inf). Found {num_nans} NaNs, "
                        f"which is {nan_percentage:.2f}% of the total values."
                    )
                    
                if self.is_regressive:
                    forecast_as_np = np.array(forecast)
                    forecast_normalized = forecast_as_np / np.abs(np.mean(forecast_as_np))
                    if self.params['adversarial'] == True:
                        adversary_forecast = -forecast_normalized
                        model_ema_performance = calculate_ema_performance(forecast_normalized, self.params['discount_factor'])
                        adversary_ema_performance = calculate_ema_performance(adversary_forecast, self.params['discount_factor'])
                        performance_aware_prediction = generate_masked_array(model_ema_performance, adversary_ema_performance, self.params['risk'])
                        profit = np.sum(performance_aware_prediction * yval)
                    else:
                        profit = np.sum(forecast_normalized * yval)  # Calculate absolute errors
                else:
                    # forecast should be array of len = num classes of probabilities of each class
                    # class values should be avg value within each class. thus we get f1c1 + f2c2 ... fncn = expected value
                    # forecast normalize contains the expected values of each forecast.
                    forecast_as_np = np.array(forecast)
                    forecast_normalized = np.dot(forecast_as_np, self.params['class_values'])
                    profit = np.sum(forecast_normalized * yval)   


                total_profit += profit
                avg_profit = total_profit / (idx + 1)

                # Log intermediate results after each block
                with open(log_file, "a") as f:
                    f.write(
                        f"Trial {trial.number}, block={idx}, "
                        f"block_profit={profit:.4f}, avg_profit={avg_profit:.4f}\n"
                    )

                trial.report(avg_profit, step=idx)

                if trial.should_prune():
                    with open(log_file, "a") as f:
                        f.write(f"Trial {trial.number} pruned at block={idx}.\n")
                    raise optuna.exceptions.TrialPruned()

            # Once done with all blocks, log final result for this trial
            with open(log_file, "a") as f:
                f.write(
                    f"=== Completed trial {trial.number} with final avg profit: {avg_profit:.4f} ===\n"
                )

            # We want to maximize profit, so return the average profit (the study is set to maximize this)
            trial.set_user_attr('params', model_params)
            return avg_profit

        # Create a new study for this tuning run.
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )

        # If we have previous tuning trials (warm start), add them to the new study
        # with their profit values halved. This “penalizes” old performance to encourage exploration.
        if self.warm_start_trials:
            from optuna.trial import create_trial  # requires optuna>=2.10.0
            for old_trial in self.warm_start_trials:
                if old_trial.value is None:
                    continue
                warm_trial = create_trial(
                    params=old_trial.params,
                    distributions=old_trial.distributions,
                    value=old_trial.value * 0.8,
                )
                # Ensure that the trial has a 'params' user attribute.
                warm_trial.user_attrs['params'] = old_trial.user_attrs.get('params', old_trial.params)
                study.add_trial(warm_trial)
                print('warm_trial done')


        # Run the study
        try:
            study.optimize(
                objective,
                n_trials=self.num_trials_per_tuning,
                n_jobs=1
            )
        except Exception as e:
            print("Exception encountered:", e)
            with open(log_file, "a") as f:
                f.write(f"Exception encountered during Optuna study: {str(e)}\n")

        # Best trial results
        best_trial = study.best_trial
        self.params = best_trial.user_attrs['params']
        self.best_profit = best_trial.value  # store best profit

        # Initialize the model with the best hyperparameters
        prepare_data(self)
        self.initialize_model(self.params)

        # Log the best trial info
        with open(log_file, "a") as f:
            f.write(f"\n=== Best Trial Summary ===\n")
            f.write(f"Trial number: {best_trial.number}\n")
            f.write(f"Best parameters: {self.params}\n")
            f.write(f"Best profit: {self.best_profit}\n")

        # Update warm start trials for future tuning runs.
        # Here we simply save all trials from the current study.
        self.warm_start_trials = study.trials
