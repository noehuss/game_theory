from market_clearing import MarketClearing
from strategic_behavior import MPEC, MPEClinearized
import pandas as pd
import utils
from best_response import BR

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60],       
    'minFuelCosts': [10, 30, 35, 55],      
    'maxFuelCosts': [15, 60, 70, 90] 
})

demand = 155

## Exercice 2
#Truthful competitors

strategic_producer = 'P1'
mc = MarketClearing(bids=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), demand=demand, prod_df=prod_df)
print(mc.get_price())

# mpec = MPEC(producer=strategic_producer, alphas=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=100)
# print(mpec.get_profit())
# print(mpec.get_results())

# utils.plot_merit_order(mpec.get_results(), demand=demand, strategic_producer=strategic_producer)

mpec = MPEClinearized(producer='P1', alphas=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=200, upper_bid=200)
print(mpec.get_profit())
print(mpec.get_results())
profit = mpec.get_profit()

utils.plot_merit_order(mpec.get_results(), demand=demand, strategic_producer=strategic_producer, sp_profit=profit)

#Overbidding competitors

strategic_producer = 'P1'
mc = MarketClearing(bids=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), demand=demand, prod_df=prod_df)
print(mc.get_price())

mpec = MPEClinearized(producer='P1', alphas=[10, 40, 45, 120], marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=200, upper_bid=200)
print(mpec.get_profit())
print(mpec.get_results())
profit = mpec.get_profit()

utils.plot_merit_order(mpec.get_results(), demand=demand, strategic_producer=strategic_producer, sp_profit=profit)