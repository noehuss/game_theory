from market_clearing import MarketClearing
import pandas as pd

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3'],
    'capacities': [100, 40, 40],       
    'minFuelCosts': [10, 30, 35],      
    'maxFuelCosts': [15, 60, 70] 
})

demand = 150
mc = MarketClearing(bids=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), demand=demand, prod_df=prod_df)
print(mc.get_price())