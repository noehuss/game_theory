from strategic_behavior import MPEClinearized, MPEC
from market_clearing import MarketClearing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BR():
    def __init__(self, bids_init: list, marginal_costs:list, demand:int, prod_df:pd.DataFrame, tolerance:float=0.02):
        self.marginal_costs = marginal_costs
        self.alphas_init = bids_init
        self.demand = demand
        self.prod_df = prod_df

        self.iteration = 0
        self.dict_alphas = {0: bids_init}
        self.prices = []
        self.dict_profits = {}
        self.dict_dispatch = {}
        
        self.increase_profit = {producer: [] for producer in self.prod_df['producers'].values.tolist()}
        self.estimated_profit = {producer: [] for producer in self.prod_df['producers'].values.tolist()}

        self.tol = tolerance

    def run_BR(self, nb_iter:int=10):
        """
        Run the best response algorithm. Return a dataframe with 
        the converged equilibrium dispatch and a boolean to indicate if 
        the equilibrium was reached or not.
        """
        self.nb_iter = nb_iter
        while self.iteration <= nb_iter and not self.convergenceReached():
            self.iteration += 1
            self.dict_alphas[self.iteration]=self.dict_alphas[self.iteration-1].copy()
            for index, producer in enumerate(self.prod_df['producers'].values.tolist()):            
                print(f"Strategic producer {producer}")
                # Calculation of the profit before strategic decision
                mc = MarketClearing(bids=self.dict_alphas[self.iteration].copy(), 
                                marginal_costs=self.marginal_costs, 
                                demand=self.demand, 
                                prod_df=self.prod_df)
                profit_bf_strategic_decision = mc.get_profits()[index]
                mpec = MPEClinearized(producer=producer, 
                                      alphas=self.dict_alphas[self.iteration].copy(), 
                                      marginal_costs=self.marginal_costs, 
                                      prod_df=self.prod_df, 
                                      demand=self.demand, tau=0.5)
                self.dict_alphas[self.iteration] = mpec.update_alphas()
                self.estimated_profit[producer].append(mpec.get_profit())
                self.increase_profit[producer].append(mpec.get_profit()-profit_bf_strategic_decision)

            mc = MarketClearing(bids=self.dict_alphas[self.iteration].copy(), 
                                marginal_costs=self.marginal_costs, 
                                demand=self.demand, 
                                prod_df=self.prod_df)
            self.prices.append(mc.get_price())
            self.dict_profits[self.iteration] = mc.get_profits()
            self.dict_dispatch[self.iteration] = mc.get_dispatch()

        if not self.convergenceReached():
            self.iteration += 1
            self.prices.append(sum(self.prices[-10:-1])/10)
            profit_last_iter = [self.dict_profits[k] for k in range(self.iteration-1, self.iteration-10, -1)]
            dispatch_last_iter = [self.dict_dispatch[k] for k in range(self.iteration-1, self.iteration-10, -1)]
            alphas_last_iter = [self.dict_alphas[k] for k in range(self.iteration-1, self.iteration-10, -1)]
            self.dict_profits[self.iteration] = np.array([sum(i) for i in zip(*profit_last_iter)])/10
            self.dict_dispatch[self.iteration] = np.array([sum(i) for i in zip(*dispatch_last_iter)])/10
            self.dict_alphas[self.iteration] = np.array([sum(i) for i in zip(*alphas_last_iter)])/10 #type: ignore
            print(self.dict_alphas)


    def convergenceReached(self) -> bool:
        if self.iteration in [0, 1] or self.iteration > self.nb_iter:
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
    
    def plot_estimated_profits_evo(self) -> None:
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot estimated profits
        df_estimated = pd.DataFrame(self.estimated_profit)
        df_estimated.plot(ax=ax1)
        ax1.set_title('Estimated Profits Evolution')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Profit')
        ax1.grid(True)
        ax1.legend(title='Producer')

        # Create DataFrame for market clearing profits
        market_profits = {producer: [] for producer in self.prod_df['producers'].values.tolist()}
        for i in range(1, self.iteration + 1):
            for idx, producer in enumerate(self.prod_df['producers'].values.tolist()):
                market_profits[producer].append(self.dict_profits[i][idx])

        df_market = pd.DataFrame(market_profits)
        df_market.plot(ax=ax2)
        ax2.set_title('Market Clearing Profits Evolution')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Profit')
        ax2.grid(True)
        ax2.legend(title='Producer')

        plt.tight_layout()
        plt.show()
        
    def plot_strategic_behaviour(self):
        """
        Plot expected profit before and after strategic decision
        """
        fig, ax = plt.subplots(figsize=(15, 5))

        # Plot estimated profits
        df = pd.DataFrame(self.increase_profit)
        df.plot(ax=ax, marker='o')
        ax.set_title('Variation of profit, before and after strategic decision')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('$\Delta$ Profit')
        ax.grid(True)
        ax.legend(title='Producer')
        plt.show()