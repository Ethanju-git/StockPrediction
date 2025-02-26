from collections import deque
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from utils.data_processing import prepare_data
from datetime import datetime
import os
import torch


def walk_forward_validation(model):
    # 1) Initial setup and tuning
    model.log_file = create_runtime_log_file(model)
    tune_model(model)
    
    # 2) Walk forward loop
    while len(model.unseen_y) > 0:
        #Tune if necessary
        if should_retune_model(model):
            tune_model(model)
            model.steps_since_tuned = 0
            
        #Get the prediction and convert it to a continuous value if necessary
        if model.is_regressive:
            pred = model.predict_next_step(model.seen_X)
        else:
            pred = model.predict_next_step(model.seen_X)
            
            if torch.is_tensor(pred):
                pred = pred.detach().cpu().numpy()
            elif isinstance(pred, list):
                pred = np.array([item.detach().cpu().numpy() if torch.is_tensor(item) else item for item in pred])
            elif isinstance(pred, dict):
                pred = {key: (value.detach().cpu().numpy() if torch.is_tensor(value) else value) for key, value in pred.items()}
            else:
                pred = np.array(pred)
            
            pred = np.array(pred)
            pred = np.dot(pred, model.params['class_values'])

        #Walk forward
        actual = model.unseen_y[:1]
        model.unseen_y = model.unseen_y[1:]
        model.seen_y = np.append(model.seen_y, actual)
        model.seen_X = np.append(model.seen_X, model.unseen_X[:1], axis=0)
        model.unseen_X = model.unseen_X[1:]

        #Update cumulative predictions, actuals
        model.total_predictions = (
            np.array([pred]) if model.total_predictions is None or model.total_predictions.size == 0
            else np.concatenate((model.total_predictions, np.array([pred])))
        )
        
        model.total_actuals = (
            np.array([actual]) if model.total_actuals is None or model.total_actuals.size == 0
            else np.concatenate((model.total_actuals, np.array([actual])))
        )
        
        #Use competing model instead based on recent performance? Currently simulate via a model that inverts our preds
        if model.params['adversarial'] == True:
            pred_len = len(model.total_predictions)
            sequence = np.power(model.params['discount_factor'], np.arange(pred_len))[::-1]
            pred_ema = np.sum((sequence * model.total_predictions) * model.total_actuals)

            adversary_ema = -pred_ema  #for now single model implementation

            if pred_ema > 0 and adversary_ema < 0:
                pass #keep the predicted value the same
            elif pred_ema > 0 and adversary_ema > 0:
                if model.params['risk'] == 'Conservative':
                    model.total_predictions[-1] = 0
                elif model.params['risk'] == 'Liberal':
                    pass
            elif pred_ema < 0 and adversary_ema < 0:
                if model.params['risk'] == 'Conservative':
                    model.total_predictions[-1] = 0
                elif model.params['risk'] == 'Liberal':
                    model.total_predictions[-1] = -model.total_predictions[-1]
            elif pred_ema < 0 and adversary_ema > 0:
                model.total_predictions[-1] = -model.total_predictions[-1]

        
        model.steps_since_tuned += 1
    log_results(model)





def tune_model(model):
    model.tune_hyperparameters()
    model.fit(model.seen_X, model.seen_y)



    

def should_retune_model(model):
    # has it been enough steps
    if model.steps_since_tuned < model.min_steps_per_retune:
        return False
    
    # calculate performance
    predictions = model.total_predictions[-model.steps_since_tuned:]
    actuals = model.total_actuals[-model.steps_since_tuned:]
    profit = 0
    benchmark = 0
    for pred, actual in zip(predictions, actuals):
        if pred > 0:
            profit += actual
        else:
            profit -= actual
        benchmark += actual
    
    max_long_short = abs(benchmark) #take max of long/short since long = -short.
    
    # should retune based on performance?
    if profit <= max_long_short:
        #add log statement
        write_to_log(model.log_file, f"Tuning triggered: performance less than benchmark. Steps since tuned: {model.steps_since_tuned}")
        return True
    else:
        write_to_log(model.log_file, f'Tuning not triggered: performance exceeds benchmark. Steps since tuned: {model.steps_since_tuned}')
        return False
        


def create_runtime_log_file(model):
    """
    Create a unique log file for the current runtime.
    """
    os.makedirs('logs', exist_ok=True)
    criteria = datetime.now().strftime(f'{model.target_ticker}_{model.start_date}_{model.end_date}_{model.__class__.__name__}_{model.num_trials_per_tuning}_{model.min_steps_per_retune}')
    log_file = f"logs/walk_forward_validation_{criteria}.log"
    return log_file

def write_to_log(log_file, message):
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"[{timestamp}] {message}\n")
        
def log_results(model):

    actuals = np.ravel(model.total_actuals)
    predictions = np.ravel(model.total_predictions)
    os.makedirs('results', exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"results/{current_time}.csv"

    # Create a DataFrame with actuals and predictions
    results_df = pd.DataFrame({
        'Actuals': actuals,
        'Predictions': predictions
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv(file_name)
 