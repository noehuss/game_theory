from strategic_behavior import MPEClinearized, MPEC
from market_clearing import MarketClearing
import pandas as pd
import numpy as np

class BR():
    def __init__(self, bids_init: list, marginal_costs:list, demand:int, prod_df:pd.DataFrame):
        self.marginal_costs = marginal_costs
        self.alphas_init = bids_init
        self.demand = demand
        self.prod_df = prod_df

        self.iteration = 0
        self.dict_alphas = {0: bids_init}
        self.prices = []
        self.dict_profits = {}
        self.dict_dispatch = {}

        self.tol = 0.01

    def run_BR(self, nb_iter:int=10):
        """
        Run the best response algorithm. Return a dataframe with 
        the converged equilibrium dispatch and a boolean to indicate if 
        the equilibrium was reached or not.
        """
        while self.iteration <= nb_iter and not self.convergenceReached():
            self.iteration += 1
            self.dict_alphas[self.iteration]=self.dict_alphas[self.iteration-1].copy()
            for producer in self.prod_df['producers'].values.tolist():            
                print(f"Strategic producer {producer}")
                mpec = MPEClinearized(producer=producer, 
                                      alphas=self.dict_alphas[self.iteration].copy(), 
                                      marginal_costs=self.marginal_costs, 
                                      prod_df=self.prod_df, 
                                      demand=self.demand)
                self.dict_alphas[self.iteration] = mpec.update_alphas()
            mc = MarketClearing(bids=self.dict_alphas[self.iteration].copy(), 
                                marginal_costs=self.marginal_costs, 
                                demand=self.demand, 
                                prod_df=self.prod_df)
            self.prices.append(mc.get_price())
            self.dict_profits[self.iteration] = mc.get_profits()
            self.dict_dispatch[self.iteration] = mc.get_dispatch()

    def convergenceReached(self) -> bool:
        if self.iteration in [0, 1]:
            return False
        return np.allclose(np.array(self.dict_profits[self.iteration]), 
                           np.array(self.dict_profits[self.iteration-1]), 
                           rtol=self.tol)
    
    def get_results(self) -> tuple[pd.DataFrame, bool]:
        data = {
            'production':  self.dict_dispatch[self.iteration],
            'bids': self.dict_alphas[self.iteration],
            'producer': self.prod_df['producers'].to_list(),
            'capacities': self.prod_df['capacities'].to_list()
        }

        return pd.DataFrame(data), self.convergenceReached()
    
    def get_price(self) -> float:
        return self.prices[-1]