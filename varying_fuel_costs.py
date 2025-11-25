import pandas as pd
import numpy as np
from itertools import product
from best_response import BR
from market_clearing import MarketClearing
import matplotlib.pyplot as plt


# prod_df = pd.DataFrame({
#     'producers': ['P1', 'P2', 'P3', 'P4'],
#     'capacities': [40, 90, 50, 60], 
#     'Pmax': [40, 90, 50, 60],       
#     'Pmin': [10, 0, 30, 20],      
#     'minFuelCosts': [10, 30, 35, 55],      
#     'maxFuelCosts': [15, 60, 70, 90]
# })

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'minFuelCosts': [10, 30, 35, 55],      
    'maxFuelCosts': [15, 60, 70, 90]
})

demand = 155

def search_worst_inefficiency(prod_df: pd.DataFrame, demand:int, nb_segments:int=3) -> float:
    """
    Iterate over a bounded discrete grid of marginal costs. 
    """
    nb_producers = len(prod_df['producers'])

    segments = []
    thetas = []
    for i in range(nb_producers):
        rangeFuel = np.linspace(prod_df['minFuelCosts'][i], prod_df['maxFuelCosts'][i], num=nb_segments, dtype=int)
        segments.append(list(rangeFuel))
    for theta in (product(*segments)):
        thetas.append(list(theta))
    print(thetas)
    list_PoA = []
    list_convergence = []
    for theta in thetas:
        # Best response algorithm
        br = BR(bids_init=theta, marginal_costs=theta, demand=demand, prod_df=prod_df)
        br.run_BR(100)
        list_convergence.append(br.convergenceReached())
        # Market clearing
        q_eq = np.array(br.get_equilibrium_dispatch(), dtype=float)
        theta_arr = np.array(theta, dtype=float)
        SC_eq = float(np.dot(theta_arr, q_eq))
        # Verify that the equilibrium dispatch meets the demand
        mc_opt = MarketClearing(bids = theta, marginal_costs=theta, demand=demand, prod_df=prod_df)
        q_opt = np.array(mc_opt.get_dispatch(), dtype=float)
        SC_opt = float(np.dot(theta_arr, q_opt))    
        # PoA
        PoA = SC_eq / SC_opt
        #mc = MarketClearing(bids = theta, marginal_costs=theta, demand=demand, prod_df=prod_df)
        # PoA
        #   PoA = br.get_price()/mc.get_price()
        print(PoA)
        list_PoA.append(PoA)

    print(thetas)
    print(list_PoA)
    print(f"Share of instances that converged: {sum(list_convergence)/len(list_convergence):.2f}")
    plt.hist(list_PoA, bins=20)
    plt.show()

    return max(list_PoA)


if __name__ == "__main__":
    worst_inefficiency = search_worst_inefficiency(prod_df=prod_df, demand=demand, nb_segments=3)
    print(f"Worst inefficiency (PoA) found: {worst_inefficiency:.2f}")
# def inefficiencies(nb_players, capacity, c_lower, c_upper, nb_segments:int=2):
#     prod_df = pd.DataFrame(columns=['producers', 'capacities', 'minFuelCosts', 'maxFuelCosts'])
#     list_inefficiencies = []
#     for i in range(nb_players+1):
#         prod_df = pd.concat([pd.DataFrame([[f'P{i}', capacity, c_lower, c_upper]], columns=prod_df.columns),  prod_df], ignore_index=True)
#         demand = prod_df['capacities'].sum()*0.55 
#         if i < 3:
#             continue
#         list_inefficiencies.append(search_worst_inefficiency(prod_df=prod_df, demand=demand, nb_segments=nb_segments))
#     return list_inefficiencies

# print(inefficiencies(5, 100, 10, 50, nb_segments=3))
