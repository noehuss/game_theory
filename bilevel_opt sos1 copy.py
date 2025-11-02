from pyomo.environ import * #type: ignore
import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
# data

#prod_df = pd.DataFrame(data={'producers':['P1', 'P2', 'P3', 'P4'], 'capacities':[50, 40, 50, 60], 
#                             'minFuelCosts':[30, 2, 50, 20], 'maxFuelCosts':[100, 5, 55, 30]})
prod_df = pd.DataFrame({
    'producers': ['P1', 'P2', 'P3'],
    'capacities': [100, 40, 40],       # P1 domine
    'minFuelCosts': [10, 30, 35],      # P1 est beaucoup plus efficace
    'maxFuelCosts': [15, 60, 70]       # P1 a un intervalle faible
})

demand = 120  # MW

# prod_df = pd.DataFrame({
#     'producers': ['P1', 'P2', 'P3'],
#     'capacities': [60, 60, 60],        # Tous ont assez de capacité
#     'minFuelCosts': [20, 22, 21],      # Coûts très proches
#     'maxFuelCosts': [25, 27, 26]       # Légère incertitude
# })

# demand = 100  # MW

# prod_df = pd.DataFrame({
#     'producers': ['P1', 'P2', 'P3', 'P4'],
#     'capacities': [30, 25, 20, 15],    # Total = 90 MW
#     'minFuelCosts': [15, 20, 35, 45],
#     'maxFuelCosts': [25, 30, 50, 60]
# })

# demand = 85  # MW (très proche du total)
# model
# model = ConcreteModel()

# # index
# nb_producers = len(prod_df['producers'])
# nb_segments = 2

# segments = []
# for i in range(nb_producers):
#     rangeFuel = np.linspace(prod_df['minFuelCosts'][i], prod_df['maxFuelCosts'][i], num=nb_segments, dtype=int)
#     segments.append(list(rangeFuel))
# thetas =list(product(segments[0], segments[1], segments[2]))
# print(thetas)
# model.IndexThetas = RangeSet(0, nb_segments**nb_producers-1)

# def rule_theta(model, i):
#     return thetas[i]

# model.theta = Param(model.IndexThetas, initialize=[list(thetas[i]) for i in range(len(thetas))], within=Any)

# print(model.theta[1][2]) #type: ignore

class MPEC():
    
    def __init__(self, producer_name, alphas, marginal_costs, demand):
        self.model = ConcreteModel()
        self.name = producer_name 
        self.alphas = alphas
        self.marginal_costs = marginal_costs.copy()
        self.tau = 1
        self.bigM = 1000
        self.demand = demand
        self.indexes()
        self.parameters()
        self.variables(upper_bid=100, lower_bid=0)
        self.constraints()
        self.objective_function()
        self.solve()
        self.update_alphas()

    def indexes(self):    
        self.model.Omega = Set(initialize = prod_df[prod_df['producers']==self.name].index.tolist())
        self.model.minusOmega = Set(initialize = prod_df[prod_df['producers']!=self.name].index.tolist())
        self.model.I = Set(initialize = prod_df.index.tolist())

    def parameters(self):
        self.model.capacities = Param(self.model.I, initialize=prod_df['capacities'].tolist())
        self.model.alphas = Param(self.model.I, initialize=self.alphas)

    def variables(self, upper_bid, lower_bid):
        # variables
        self.model.alpha_g = Var(self.model.Omega, domain=NonNegativeReals, bounds=(lower_bid, upper_bid))
        self.model.mu_max = Var(self.model.I, domain=NonNegativeReals)
        self.model.mu_min = Var(self.model.I, domain=NonNegativeReals)
        self.model.price = Var(domain=Reals)
        self.model.Pg = Var(self.model.I,  domain=NonNegativeReals)
        # BigM variables
        self.model.z_max = Var(self.model.I, domain=Binary)
        self.model.z_min = Var(self.model.I, domain=Binary)
        # Differences alphas
        self.model.y = Var(self.model.minusOmega, domain=Binary)

    def constraints(self):
        # constraints
        def rule_price_balance(model, i):
            if i in model.Omega:
                return model.alpha_g[i] + model.mu_max[i] - model.mu_min[i] - model.price == 0
            else:
                return model.alphas[i] + model.mu_max[i] - model.mu_min[i] - model.price == 0
            
        self.model.constraint_price_balance = Constraint(self.model.I, rule=rule_price_balance)

        self.model.constraint_power_balance = Constraint(rule=(self.demand-sum([self.model.Pg[i]for i in self.model.I]) == 0)) #type: ignore

        # Alpha constraint
        def rule_in_up(model, i, j):
            return model.alpha_g[i] <= (model.alphas[j] - self.tau) + self.bigM * (model.y[j]) # type: ignore
        self.model.constraint_in_up_alpha_g = Constraint(self.model.Omega, self.model.minusOmega, rule=rule_in_up)

        def rule_in_down(model, i, j):
            return model.alpha_g[i] >= (model.alphas[j] + self.tau) - self.bigM *( 1- model.y[j]) # type: ignore
        self.model.constraint_in_down_alpha_g = Constraint(self.model.Omega, self.model.minusOmega, rule=rule_in_down)
        self.model.constraint_in_down_alpha_g.display()
        
        # def rule_aplha_mc(model,i):
        #     return model.alpha_g[i] >= self.marginal_costs[i]
        # self.model.constraint_alpha_mc = Constraint(self.model.Omega, rule=rule_aplha_mc)

        # Complementary slackness constraints
        def rule_pgmax(model, i):
            return model.capacities[i] - model.Pg[i] <= self.bigM*model.z_max[i]
        self.model.constraint_pgmax = Constraint(self.model.I, rule=rule_pgmax)

        def rule_capamax(model, i):
            return model.Pg[i] <= model.capacities[i]
        self.model.constraint_capamax = Constraint(self.model.I, rule=rule_capamax)

        def rule_mu_max(model, i):
            return model.mu_max[i] <= self.bigM*(1-model.z_max[i])
        self.model.constraint_mu_max = Constraint(self.model.I, rule=rule_mu_max)

        def rule_pgmin(model, i):
            return model.Pg[i] <= self.bigM*model.z_min[i]
        self.model.constraint_pgmin= Constraint(self.model.I, rule=rule_pgmin)

        def rule_mu_min(model, i):
            return model.mu_min[i] <= self.bigM*(1-model.z_min[i])
        self.model.constraint_mu_min = Constraint(self.model.I, rule=rule_mu_min)

    def objective_function(self):
        # objective function
        self.model.obj = Objective(expr=sum([-self.model.price*self.model.Pg[i] + self.marginal_costs[i]*self.model.Pg[i] for i in self.model.Omega]), sense=minimize) #type: ignore

    def solve(self):
        # Dual
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        # Create a solver
        solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured
        # Solve the model
        solution = solver.solve(self.model, tee=True)
        self.model.Pg.display()
        self.model.price.display()
        self.model.alpha_g.display()
        self.model.constraint_price_balance.display()
        self.model.constraint_in_up_alpha_g.display()
        self.model.constraint_in_down_alpha_g.display()
        self.model.y.display()


    def get_profit(self):
        return - value(self.model.obj) # type: ignore
    
    def get_price(self):
        return value(self.model.price)
    
    def update_alphas(self):
        for i in prod_df[prod_df['producers']==self.name].index.tolist():
            self.alphas[i] = value(self.model.alpha_g[i]) #type: ignore



class BR():
    def __init__(self, marginal_costs:list, demand:int):
        self.marginal_costs = marginal_costs
        self.alphas_init  = marginal_costs
        self.alphas_list = [self.alphas_init.copy()]
        self.alphas_dict = {
            0: self.alphas_init.copy(),
        }
        self.profits = pd.DataFrame()
        self.iteration = 0
        self.delta_profits = pd.DataFrame()
        self.demand = demand
        self.price = 0

    def run_BR(self, nb_iteration):
        print(self.alphas_list)
        self.prices = []
        self.delta_profits[0] = pd.Series(data=[1 for i in range(len(prod_df))], index=prod_df['producers'].values.tolist())
        self.delta_profits[1] = pd.Series(data=[1 for i in range(len(prod_df))], index=prod_df['producers'].values.tolist())
        while self.iteration <= nb_iteration and self.test_convergence(self.delta_profits[self.iteration], threshold=0.001):
            self.iteration += 1
            self.profits[self.iteration] = pd.Series()
            for producer in prod_df['producers'].values.tolist():
                print(producer)
                mpec = MPEC(producer, alphas=self.alphas_list[-1].copy(), marginal_costs=self.marginal_costs.copy(), demand=self.demand)
                mpec.model.pprint()
                self.profits.loc[producer, self.iteration] = mpec.get_profit()
                self.alphas_list.append(mpec.alphas.copy())
                self.alphas_dict[self.iteration] = mpec.alphas
            mc = MarketClearing(self.alphas_dict[self.iteration], demand=self.demand, true_costs=self.marginal_costs.copy())
            self.profits[self.iteration] = np.array(mc.get_profits())
            self.price = mc.get_price()
            self.prices.append(self.price)
            if self.iteration > 1:
                for producer in prod_df['producers'].values.tolist():
                    previous_profit =  self.profits.loc[producer, self.iteration-1]
                    self.delta_profits.loc[producer, self.iteration] = abs((self.profits.loc[producer, self.iteration] - previous_profit )/ previous_profit) if  previous_profit !=0 else 0 #type: ignore
        

    def get_price(self) -> float:
        return self.price #type: ignore

    def test_convergence(self, delta:pd.Series, threshold):
        for element in delta.values.tolist(): #type: ignore
            if element >= threshold or element is None:
                return True
        return False

class MarketClearing:
    def __init__(self, marginal_costs:list, demand:int, true_costs:list):
        self.model = ConcreteModel()
        self.demand = demand
        self.marginal_costs = marginal_costs
        self.true_marginal_costs = true_costs
        self.indexes()
        self.parameters()
        self.variables()
        self.constraints()
        self.objective_function()
        self.solve()

    def indexes(self):
        self.model.G = Set(initialize = prod_df.index.tolist())

    def parameters(self):
        self.model.capacities = Param(self.model.G, initialize = prod_df['capacities'].tolist())
        self.model.marginal_costs = Param(self.model.G, initialize = self.marginal_costs)

    def variables(self):
        self.model.Pg = Var(self.model.G, domain=NonNegativeReals)

    def constraints(self):
        def rule_prod(model, g):
            return model.Pg[g] <= model.capacities[g]
        self.model.prod_constraint = Constraint(self.model.G, rule= rule_prod)

        self.model.eq_constraint = Constraint(rule = sum(self.model.Pg[g] for g in self.model.G) == self.demand) # type: ignore


    def objective_function(self):
        # objective function
        self.model.obj = Objective(expr = sum(self.model.Pg[g]*self.model.marginal_costs[g] for g in self.model.G), sense=minimize) # type: ignore

    def solve(self):
        # Dual
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        # Create a solver
        solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured
        # Solve the model
        solution = solver.solve(self.model, tee=True)
        self.model.Pg.display()
        self.model.dual.display()

    def get_price(self) -> float:
        return value(self.model.dual[self.model.eq_constraint]) #type: ignore
    
    def get_profits(self):
        return [value(self.model.Pg[g])*self.get_price() - value(self.model.Pg[g])*value(self.true_marginal_costs[g]) for g in self.model.G] # type: ignore



# # Implement market clearing
# mc = MarketClearing(marginal_costs=prod_df['minFuelCosts'].tolist(), demand=demand, true_costs=prod_df['minFuelCosts'].tolist())

# # Implement PoA calculation
# print(br.get_price())
# PoA = br.get_price()/mc.get_price()
# print(PoA)
# # Iterate BR over the set of costs
# test = pd.DataFrame(br.alphas_dict).transpose()

# fig, ax = plt.subplots()
# ax.plot(br.prices)
# test.plot(ax=ax)
# plt.show()

# index
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
    br = BR(marginal_costs=theta, demand=demand)
    br.run_BR(50)
    # Market clearing
    mc = MarketClearing(marginal_costs=theta, demand=demand, true_costs=theta)
    # fig, ax = plt.subplots()
    # ax.plot(br.profits.transpose())
    #test.plot(ax=ax)
    plt.show()
    # PoA
    PoA = br.get_price()/mc.get_price()
    print(PoA)
    list_PoA.append(PoA)

print(list_PoA)
plt.hist(list_PoA, bins=10)
plt.show()