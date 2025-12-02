# run_multi_period.py

import pandas as pd
from multi_period_br import MultiPeriodBR
from plot_results import plot_results


if __name__ == "__main__":
    # Example producer data
    prod_df = pd.DataFrame({
        'producers': ['P1', 'P2', 'P3', 'P4'],
        'capacities': [40, 90, 50, 60], 
        'Pmax':       [40, 90, 50, 60],
        'Pmin':       [10,  0, 30, 20],
        'minFuelCosts': [10, 30, 35, 55],
        'maxFuelCosts': [15, 60, 70, 90],
        'rampingRate':  [10, 30, 15, 20],
    })

    # Marginal costs (you can use minFuelCosts)
    marginal_costs = prod_df['minFuelCosts'].tolist()

    # Demand per time step (5 periods)
    demand_ts = [120, 130, 110, 140, 125]

    # Initial bids: start from marginal costs
    bids_init = marginal_costs[:]

    mp_br = MultiPeriodBR(
        prod_df=prod_df,
        marginal_costs=marginal_costs,
        demand_ts=demand_ts,
        bids_init=bids_init,
        tolerance=0.02,
    )

    mp_br.run(nb_iter_per_t=200)

    df_all = mp_br.get_bids_time()
    print("\nAll time-step results:")
    print(df_all)

    plot_results(mp_br, prod_df, marginal_costs)