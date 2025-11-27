from best_response import BR
import pandas as pd
import numpy as np

def update_prof_df(prod_df:pd.DataFrame, pg:pd.Series):
    df = prod_df.set_index('producers')
    prod_df['Pmax'] = pd.concat([df['ramp_constraints']+pg, df['capacities']], axis=1).min(axis=1).to_list()
    prod_df['Pmin'] = pd.concat([-df['ramp_constraints']+pg, pg*0], axis=1).max(axis=1).to_list()
    return prod_df

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp_constraints': [10, 10, 10, 60] #to change
})

marginal_costs = prod_df['marginal_costs'].to_list()
net_loads = 155+0*np.random.rand(5)
print(net_loads)

results_prod = pd.DataFrame()
results_prod.index = prod_df['producers'] #type: ignore

for (hour, net_load) in enumerate(net_loads):
    br = BR(bids_init=marginal_costs, marginal_costs=marginal_costs, demand=net_load, prod_df=prod_df)
    br.run_BR(nb_iter=200, tau_alphas=1)
    results, convergence = br.get_results()
    prod_df = update_prof_df(prod_df=prod_df, pg=results.set_index('producer')['production'])

    # Store prod results
    results_prod[hour] = results.set_index('producer')['production']

    # Calculation Inefficiency
print(net_loads)
print(results_prod)
print(prod_df)
