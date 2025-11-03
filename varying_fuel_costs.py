import pandas as pd
import numpy as np
from itertools import product
from best_response import BR
from market_clearing import MarketClearing
import matplotlib.pyplot as plt

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3'],
    'capacities': [50, 40, 40],       
    'minFuelCosts': [10, 30, 35],      
    'maxFuelCosts': [15, 60, 70] 
})

demand = 92

nb_producers = len(prod_df['producers'])
nb_segments = 2

segments = []
thetas = []
for i in range(nb_producers):
    rangeFuel = np.linspace(prod_df['minFuelCosts'][i], prod_df['maxFuelCosts'][i], num=nb_segments, dtype=int)
    segments.append(list(rangeFuel))
for theta in (product(*segments)):
    thetas.append(list(theta))
print(thetas)
list_PoA = []
for theta in thetas:
    # Best response algorithm
    br = BR(bids_init=theta, marginal_costs=theta, demand=demand, prod_df=prod_df)
    br.run_BR(50)
    # Market clearing
    mc = MarketClearing(bids = theta, marginal_costs=theta, demand=demand, prod_df=prod_df)
    # PoA
    PoA = br.get_price()/mc.get_price()
    print(PoA)
    list_PoA.append(PoA)

print(thetas)
print(list_PoA)
plt.hist(list_PoA, bins=10)
plt.show()