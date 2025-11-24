from best_response import BR
import pandas as pd
import numpy as np

def update_prof_df(prod_df:pd.DataFrame, pg:pd.Series):
    prod_df['Pmax'] = np.min(pg + prod_df['ramp_constraints'].to_numpy(), prod_df['capacities'].to_numpy())
    prod_df['Pmin'] = np.max(pg - prod_df['ramp_constraints'].to_numpy(), 0)
    return prod_df

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp_constraints': [10, 10, 10, 10] #to change
})

marginal_costs = prod_df['marginal_cost'].to_list()
net_loads = [155]*24

for (hour, net_load) in enumerate(net_loads):
    br = BR(bids_init=marginal_costs, marginal_costs=marginal_costs, demand=net_load, prod_df=prod_df)
    results, convergence = br.get_results()
    prod_df = update_prof_df(prod_df=prod_df, pg=results['production'])

    # Calculation Inefficiency
    