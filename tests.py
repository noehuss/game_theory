from market_clearing import MarketClearing
from strategic_behavior import MPEC, MPEClinearized
import pandas as pd
import utils

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3'],
    'capacities': [50, 40, 40],       
    'minFuelCosts': [10, 30, 35],      
    'maxFuelCosts': [15, 60, 70] 
})

demand = 100
strategic_producer = 'P1'
mc = MarketClearing(bids=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), demand=demand, prod_df=prod_df)
print(mc.get_price())

mpec = MPEC(producer=strategic_producer, alphas=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=100)
print(mpec.get_profit())
print(mpec.construct_df())

utils.plot_merit_order(mpec.construct_df(), demand=demand, strategic_producer=strategic_producer)

mpec = MPEClinearized(producer='P1', alphas=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=100)
print(mpec.get_profit())
print(mpec.construct_df())

utils.plot_merit_order(mpec.construct_df(), demand=demand, strategic_producer=strategic_producer)