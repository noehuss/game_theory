from market_clearing_mt import MarketClearingMT
from strategic_behavior_mt import MPEC, MPEClinearized
import pandas as pd
import utils
from best_response import BR

prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3', 'P4'],
    'capacities': [40, 90, 50, 60], 
    'Pmax': [40, 90, 50, 60],       
    'Pmin': [0, 0, 0, 0],      
    'marginal_costs': [10, 30, 35, 55],
    'ramp': [10, 10, 10, 60] #to change
})

demand = [155, 175, 155]
strategic_producer = 'P1'
mc = MarketClearingMT(bids=[prod_df['marginal_costs'].to_list()]*3, marginal_costs=prod_df['marginal_costs'].to_list(), demand=demand, prod_df=prod_df)
print(mc.get_price(1))

print(mc.get_results(0))
print(mc.get_profits())
print(mc.get_social_cost(0))
mpec = MPEC(producer=strategic_producer, alphas=[prod_df['marginal_costs'].to_list()]*3, marginal_costs=prod_df['marginal_costs'].to_list(), prod_df=prod_df, demand=demand, bigM=100)
# print(mpec.get_profit())
# print(mpec.get_results())

# utils.plot_merit_order(mpec.get_results(), demand=demand, strategic_producer=strategic_producer)

# mpec = MPEClinearized(producer='P1', alphas=prod_df['minFuelCosts'].to_list(), marginal_costs=prod_df['minFuelCosts'].to_list(), prod_df=prod_df, demand=demand, bigM=100)
# print(mpec.get_profit())
# print(mpec.get_results())
# profit = mpec.get_profit()

# utils.plot_merit_order(mpec.get_results(), demand=demand, strategic_producer=strategic_producer, sp_profit=profit)


# br = BR(bids_init=[10, 30, 35, 55], marginal_costs=[10, 30, 35, 55], demand=demand, prod_df=prod_df, tolerance=0.01)
# br.run_BR(200)
# results_br, as_converged = br.get_results()
# price_br = br.get_price()
# print(br.dict_alphas)
# print(price_br)
# br.plot_estimated_profits_evo()
# br.plot_strategic_behaviour()
# # mc = MarketClearing(bids=[10, 30, 35], marginal_costs=[10, 30, 35], demand=demand, prod_df=prod_df)
# # results_mc = mc.get_results()
# # price_mc = mc.get_price()
# # #print(br.dict_alphas)
# print(utils.plot_bids_evolution(prod_df, br.dict_alphas, single_graph=False))

# utils.plot_dispatch_br_mc(results_mc=results_mc, results_br=results_br, demand=demand)
# print(f"Inefficiency of the equilibrium: {price_br}/{price_mc} = {price_br/price_mc:.2f}")
