#%%
import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt

# Data initialization
prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp': [40, 90, 15, 60] #to change
})

marginal_costs = prod_df['marginal_costs'].to_list()
demands = [155, 155, 155]

#%%
from Base.best_response import BR
from Base.market_clearing import MarketClearing

def update_prof_df(prod_df:pd.DataFrame, pg:pd.Series):
    df = prod_df.set_index('producers')
    prod_df['Pmax'] = pd.concat([df['ramp']+pg, df['capacities']], axis=1).min(axis=1).astype(int).to_list()
    prod_df['Pmin'] = pd.concat([-df['ramp']+pg, pg*0], axis=1).max(axis=1).astype(int).to_list()
    return prod_df



results_prod = pd.DataFrame()
results_prod.index = prod_df['producers'] #type: ignore
PoAs = []
fig, axes = plt.subplots(len(demands), 1, figsize=(16, 10*len(demands)), sharex=True)

for (hour, demand) in enumerate(demands):
    print(prod_df)
    br = BR(bids_init=marginal_costs, marginal_costs=marginal_costs, demand=demand, prod_df=prod_df)
    br.run_BR(nb_iter=200, tau_alphas=1)
    print(br.dict_dispatch)
    print(br.dict_alphas)
    print(br.dict_profits)
    #br.plot_estimated_profits_evo()
    #br.plot_strategic_behaviour()
    #utils.plot_bids_evolution(prod_df, br.dict_alphas, single_graph=True, market_prices=br.prices)
    results, convergence = br.get_results()
    # Store prod results
    results_prod[hour] = results.set_index('producer')['production']
    
    axes[hour] = utils.plot_merit_order(results, demand, ax_fig=axes[hour])

    # Calculation Inefficiency
    q_eq = np.array(br.get_equilibrium_dispatch(), dtype=float)
    theta_arr = prod_df['marginal_costs'].to_numpy()
    SC_eq = float(np.dot(theta_arr, q_eq))
    # Verify that the equilibrium dispatch meets the demand
    mc = MarketClearing(bids = prod_df['marginal_costs'].to_list(),
                        marginal_costs=prod_df['marginal_costs'].to_list(), 
                        demand = demand,
                        prod_df = prod_df)
    q_opt = np.array(mc.get_dispatch(), dtype=float)
    SC_opt = float(np.dot(theta_arr, q_opt))    
    # PoA
    PoA = SC_eq / SC_opt
    PoAs.append(PoA)
    prod_df = update_prof_df(prod_df=prod_df, pg=results.set_index('producer')['production'])

plt.show()
print(demands)
print(results_prod)
print(prod_df)
print(PoAs)

#%%
from MT.market_clearing_mt import MarketClearingMT
from MT.strategic_behavior_mt import MPEC, MPEClinearized
from MT.best_response_mt import BR
prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp': [40, 90, 50, 60] #to change
})

marginal_costs = prod_df['marginal_costs'].to_list()
demands = [155, 155, 155]
strategic_producer = 'P1'
print(marginal_costs)
mc = MarketClearingMT(bids=[marginal_costs.copy()]*len(demands), marginal_costs=marginal_costs.copy(), demand=demands, prod_df=prod_df)
print(mc.get_price(1))

print(mc.get_results(0))
print(mc.get_profits())
print(mc.get_social_cost(0))
mpec = MPEC(producer=strategic_producer, alphas=[marginal_costs.copy() for i in range(len(demands))] , marginal_costs=marginal_costs.copy(), prod_df=prod_df, demand=demands, bigM=1000)
print(mpec.get_profit())

fig, axes = plt.subplots(len(demands), 1, figsize=(16, 5*len(demands)), sharex=True)

for (hour, demand) in enumerate(demands):
    print(mpec.get_results(hour))
    axes[hour] = utils.plot_merit_order(mpec.get_results(hour), demand=demand, strategic_producer=strategic_producer, ax_fig=axes[hour])
plt.show()

# %%
# Without ramping constraints:
prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp': [40, 90, 50, 60] #to change
})
br = BR(bids_init=[prod_df['marginal_costs'].to_list().copy() for i in range(len(demands))], marginal_costs=[10, 30, 45, 55], demand=demands, prod_df=prod_df, tolerance=0.01, set_ramp=False)
br.run_BR(200)
results_br, as_converged = br.get_results()
price_br = br.get_price()
print("bids: ", br.dict_alphas)
print("dispatch: ", br.dict_dispatch)
print("profits: ", br.dict_profits)
print("inc profits: ", br.increase_profit)
print("price: ",price_br)
br.plot_estimated_profits_evo()
br.plot_strategic_behaviour()
# %%
# With ramping constraints:
prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 45, 55],
    'ramp': [40, 90, 15, 60] #to change
})
br = BR(bids_init=[prod_df['marginal_costs'].to_list().copy() for i in range(len(demands))], marginal_costs=[10, 30, 45, 55], demand=demands, prod_df=prod_df, tolerance=0.01, set_ramp=True)
br.run_BR(200)
results_br, as_converged = br.get_results()
price_br = br.get_price()
print("bids: ", br.dict_alphas)
print("dispatch: ", br.dict_dispatch)
print("profits: ", br.dict_profits)
print("inc profits: ", br.increase_profit)
print("price: ",price_br)
br.plot_estimated_profits_evo()
br.plot_strategic_behaviour()