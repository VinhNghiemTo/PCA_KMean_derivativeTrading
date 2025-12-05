from ropy_tools.core.alpha_manager import AlphaManager
from ropy_tools.core.final_signal import calculate_final_signals
from ropy_tools.utils.time_utils import get_min_timeframe
from ropy_tools.utils.logging_utils import logger
from dateutil.relativedelta import relativedelta
from datetime import datetime
import rework_backtrader as rbt
import os
import sys
import numpy as np
import pandas as pd
import copy
from multiprocessing import Pool
from bayes_opt import BayesianOptimization

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

# Initialize Alpha Manager
def initialize_alpha_manager(strategies_folder: str, config_path: str) -> AlphaManager:
    alpha_manager = AlphaManager(strategies_folder, config_path)
    alpha_manager.load()
    return alpha_manager

# Generate multi-strategy final signals
def generate_final_signals(alpha_manager, data_info, min_tf, selected_alphas=None):
    logger.info("Fetching data for final signal generation...")
    
    if selected_alphas:
        for alpha in alpha_manager.strategies.keys():
            alpha_manager.strategies[alpha][1]['weight'] = selected_alphas.get(alpha, 0)

    multi_signal_lists = alpha_manager.run_multi_alphas(data_info=data_info, n_workers=4, output_each_alpha=False)
    final_signals = calculate_final_signals(multi_signal_lists, min_tf)
    return final_signals, multi_signal_lists

# Run backtesting simulation
def run_simulation(
    strat_class,
    final_signals_dict,
    time_frame,
    start_date=None,
    end_date=None,
    mode=None,
    key="",
    output_flag = False ):
    simulation = rbt.RopyLab(plot=output_flag, detail_data_output=output_flag, signal_output=False)
    simulation.add_multi_api_data(
        symbols=["VN30F1M"],
        timeframe=time_frame,
        fromdate=start_date,
        todate=end_date,
        key=key,
        mode=mode,
    )
    simulation.set_cash(1e9)
    simulation.addstrategy(strat_class, final_signals_dict=final_signals_dict)
    if output_flag:
        info = simulation.run(output="alphas_results.csv",value_output=output_flag, trade_output=output_flag, log_enabled=False)
    else:
        info = simulation.run(value_output=output_flag, trade_output=output_flag, log_enabled=False)
    return info

# Define Bayesian Optimization Objective Function
def sharpe_objective(**weights):
    weight_values = np.array(list(weights.values()))
    weight_values /= np.sum(weight_values)  # Normalize to sum = 1
    
    results_copy = copy.deepcopy(global_results)
    
    # Apply the weights to each alpha
    for i, res in enumerate(results_copy):
        for entry in res:
            entry['weight'] = weight_values[i]

    final_signals_dict = calculate_final_signals(results_copy, global_min_tf)

    # Run simulation
    info = run_simulation(
        strat_class=rbt.rbtstrategies.CombinedStrategy,
        final_signals_dict=final_signals_dict,
        time_frame=global_min_tf,
        start_date=global_start_time,
        end_date=global_end_time,
        key=global_key
    )

    return -info[0].analyzers[0].get_analysis()['sharperatio']  # Maximizing Sharpe ratio

# Bayesian Optimization for optimal weights
def optimize_weights_bayesian(start_time, end_time, results, min_tf, selected_alphas, key):
    global global_results, global_min_tf, global_start_time, global_end_time, global_key
    global_results = results
    global_min_tf = min_tf
    global_start_time = start_time
    global_end_time = end_time
    global_key = key

    num_alphas = len(selected_alphas)
    pbounds = {f'w{i}': (0, 1) for i in range(num_alphas)}

    optimizer = BayesianOptimization(f=sharpe_objective, pbounds=pbounds, verbose=2, random_state=42)
    optimizer.maximize(init_points=5, n_iter=25)

    best_weights = optimizer.max['params']
    weight_values = np.array(list(best_weights.values()))
    weight_values /= np.sum(weight_values)

    optimal_weights = {alpha: weight_values[i] for i, alpha in enumerate(selected_alphas)}
    optimal_sharpe = -optimizer.max['target']

    return optimal_weights, optimal_sharpe

# Generate key rebalance dates
def generate_list_key_days(start_time, end_time):
    date_list = [start_time]
    current_date = start_time + relativedelta(months=2)

    while current_date < end_time:
        date_list.append(current_date.replace(day=1))
        date_list.append(current_date.replace(day=15))
        current_date += relativedelta(months=1)
    
    return [date for date in date_list if date < end_time]

# Compute optimal weights for a given time period
def compute_weights_for_order(order, key_dates, results, min_tf, selected_alphas, key):
    logger.info(f'Processing order {order}')
    ws, _ = optimize_weights_bayesian(key_dates[order - 1], key_dates[order], results, min_tf, selected_alphas, key)
    
    return {'ts': key_dates[order], **ws}

# Parallel update of weights
def update_weights_in_parallel(key_dates, results, min_tf, weight_table, selected_alphas, key):
    with Pool() as pool:
        orders = range(1, len(key_dates))
        result = pool.starmap(compute_weights_for_order, [(order, key_dates, results, min_tf, selected_alphas, key) for order in orders])

    for input_data in result:
        weight_table.append(input_data)

    weight_table = pd.DataFrame(weight_table)
    weight_table.ts = pd.to_datetime(weight_table.ts)
    weight_table.set_index('ts', inplace=True)
    weight_table.sort_index(inplace=True)

    # Smooth transition of weights
    for date in range(1, len(weight_table)):
        weight_table.iloc[date] = (weight_table.iloc[date] + weight_table.iloc[date - 1]) / 2

    return weight_table.reset_index()

# Main execution
if __name__ == "__main__":
    strategies_folder = "/root/futures_alphas/strategies"
    config_path = "/root/futures_alphas/configs/test_config.cfg"

    end_time = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0)
    start_time = end_time - relativedelta(months=24)
    start_time = start_time.replace(hour=9, minute=0, second=0, microsecond=0)
    key_dates = generate_list_key_days(start_time, end_time)

    api_url = "http://103.130.215.187/apivn1.php?client=nguyenluan"
    key = "a8c15e63376357f2aa70722b2f33f67f2a9d613ce2107e8f70e4c5c9286f34d1"
    data_info = (None, start_time, end_time, key)

    logger.info("Initializing Alpha Manager...")
    alpha_manager = initialize_alpha_manager(strategies_folder, config_path)
    tfs = alpha_manager.get_all_timeframes()
    min_tf = get_min_timeframe(tfs)

    logger.info("Generating final signals...")
    selected_alphas = {
        'cci_15m.CciStrategy': 0.4, 
        'ichimokucloud_20m.IchimokyCloud': 0.3,
        'mean_pivot_15m.MeanPivot': 0.3
    }

    _, results = generate_final_signals(alpha_manager, data_info, min_tf, selected_alphas)

    weight_table = [{'ts': start_time, **selected_alphas}]
    weight_table = update_weights_in_parallel(key_dates, results, min_tf, weight_table, selected_alphas, key)
    weight_table.to_csv('dynamic_weight.csv')

    final_signals_dict = calculate_final_signals(results, min_tf)
    logger.info("Running backtest simulation...")
    
    run_simulation(
        strat_class=rbt.rbtstrategies.CombinedStrategy,
        final_signals_dict=final_signals_dict,
        time_frame=min_tf,
        key=key,
        start_date=start_time,
        end_date=end_time,
        output_flag= True
    )
    
    logger.info("Simulation completed.")
