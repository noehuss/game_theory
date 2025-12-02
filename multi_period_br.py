# multi_period_br.py

import numpy as np
import pandas as pd
from best_response import BR  # your existing BR class


class MultiPeriodBR:
    """
    Wraps the single-period BR game into a T-step simulation with ramping.

    - Uses your existing BR (with MPEClinearized + MarketClearing).
    - At each time step, capacities (and Pmin) are tightened using ramp limits
      based on the previous dispatch.
    """

    def __init__(
        self,
        prod_df: pd.DataFrame,
        marginal_costs: list,
        demand_ts: list,
        bids_init: list,
        tolerance: float = 0.02,
    ):
        """
        Parameters
        ----------
        prod_df : DataFrame
            Must contain columns:
              ['producers', 'capacities', 'Pmax', 'Pmin', 'rampingRate']
        marginal_costs : list[float]
            True marginal costs per producer (same order as prod_df)
        demand_ts : list[float]
            Demand per time step, length T
        bids_init : list[float]
            Initial bids for time step 0 (same order as prod_df)
        tolerance : float
            Relative profit convergence tolerance for BR at each time step.
        """
        self.prod_df_base = prod_df.copy()
        self.marginal_costs = marginal_costs
        self.demand_ts = demand_ts
        self.bids_init = bids_init
        self.tolerance = tolerance

        self.T = len(demand_ts)
        self.G = len(prod_df)

        # Storage
        self.bids_time = []      # list of lists: bids_t[i]
        self.dispatch_time = []  # list of lists: Pg_t[i]
        self.price_time = []     # list: price_t
        self.conv_time = []      # list: BR convergence flags

    def run(self, nb_iter_per_t: int = 20):
        """
        Run the multi-period BR with ramping across all T time steps.
        """
        # Original bounds
        Pmax_orig = self.prod_df_base['Pmax'].values
        Pmin_orig = self.prod_df_base['Pmin'].values
        ramp = self.prod_df_base['rampingRate'].values

        bids_prev = self.bids_init[:]  # starting bids for t=0
        q_prev = None                  # previous dispatch

        for t in range(self.T):
            print(f"\n===== TIME STEP {t} =====")
            demand_t = self.demand_ts[t]
            print(f"Demand at time {t}: {demand_t}")
            print(f"DEBUG: demand_ts = {self.demand_ts}")

            
            # Build prod_df for this time step
            prod_df_t = self.prod_df_base.copy()

            if q_prev is not None:
                q_prev = np.array(q_prev)
                # Enforce ramping: tighten Pmax/Pmin
                Pmax_t = np.minimum(Pmax_orig, q_prev + ramp)
                Pmin_t = np.maximum(Pmin_orig, q_prev - ramp)

                prod_df_t['Pmax'] = Pmax_t
                prod_df_t['Pmin'] = Pmin_t
                # capacities used by MarketClearing = Pmax_t
                prod_df_t['capacities'] = Pmax_t
            # else t=0 → original Pmax/Pmin/capacities

            # Run BR for this time step
            demand_t = self.demand_ts[t]

            br_t = BR(
                bids_init=bids_prev,
                marginal_costs=self.marginal_costs,
                demand=demand_t,
                prod_df=prod_df_t,
                tolerance=self.tolerance,
            )

            br_t.run_BR(nb_iter=nb_iter_per_t)

            df_res_t, conv_t = br_t.get_results()
            price_t = br_t.get_price()
            print("DEBUG: BR-equilibrium dispatch:", df_res_t['production'].tolist())
            print("DEBUG: Sum dispatch =", sum(df_res_t['production']))
            print("DEBUG: Price =", price_t)
            print(f"  Converged: {conv_t}, price = {price_t:.2f}")
            print(df_res_t)

            # Store results
            self.bids_time.append(df_res_t['bids'].tolist())
            self.dispatch_time.append(df_res_t['production'].tolist())
            self.price_time.append(price_t)
            self.conv_time.append(conv_t)

            # Prepare for next time step
            bids_prev = df_res_t['bids'].tolist()
            q_prev = df_res_t['production'].tolist()

    # Convenience getters ------------------------------------------------
    def get_bids_time(self):
        # returns a DataFrame with one row per (t, producer)
        records = []
        for t in range(self.T):
            for i, prod in enumerate(self.prod_df_base['producers']):
                records.append({
                    'time': t,
                    'producer': prod,
                    'bid': self.bids_time[t][i],
                    'dispatch': self.dispatch_time[t][i],
                    'price': self.price_time[t],
                })
        return pd.DataFrame(records)

    def get_prices(self):
        return self.price_time

    def get_dispatch_time(self):
        # list over t: each entry is list of Pg[i] for that t
        return self.dispatch_time

    def get_convergence_flags(self):
        return self.conv_time
