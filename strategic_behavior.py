# Strategic behavior for one producer
from pyomo.environ import * #type: ignore
import pandas as pd

class MPEC:
    def __init__(self, producer: str, alphas: list, marginal_costs: list, prod_df:pd.DataFrame, demand: int, bigM=1e3, tau=1, upper_bid=100, lower_bid=0):
        """
        MPEC algorithm
        producer: producer name()
        alphas: current bids vector (list) for all producers
        marginal cost: marginal costs of producers
        """
        self.model = ConcreteModel()
        self.demand = demand
        self.alphas = alphas
        self.producer = producer
        self.bigM = bigM
        self.tau = tau
        self.marginal_costs = marginal_costs
        self.upper_bid = upper_bid
        self.lower_bid = lower_bid
        self.prod_df = prod_df
        # self.indexes()
        # self.parameters()
        # self.variables()
        # self.constraints()
        # self.objective_function()
        # self.solve()

    def indexes(self):
        self.model.Omega = Set(initialize = self.prod_df[self.prod_df['producers']==self.producer].index.tolist())
        self.model.OmegaBar = Set(initialize = self.prod_df[self.prod_df['producers']!=self.producer].index.tolist())
        self.model.I = Set(initialize = self.prod_df.index.tolist())

    def parameters(self):
        self.model.capacities = Param(self.model.G, initialize = self.prod_df['capacities'].tolist())
        self.model.aplhas = Param(self.model.G, initialize = self.alphas)

    def variables(self):
        # Production
        self.model.Pg = Var(self.model.I, domain=NonNegativeReals)

        # Bids
        self.model.alpha_g = Var(self.model.Omega, domain=NonNegativeReals, bounds=(lower_bid, upper_bid))
                
        self.model.price = Var(domain=Reals)

        self.model.mu_max = Var(self.model.I, domain=NonNegativeReals)
        self.model.mu_min = Var(self.model.I, domain=NonNegativeReals)

        # Binary variables
        self.model.z_max = Var(self.model.I, domain = Binary)
        self.model.z_min = Var(self.model.I, domain = Binary)

        # Alphas
        self.model.y_diff = Var(self.model.OmegaBar, domain=Binary)

    def constraints(self):
        # Price and power balance
        def rule_price_balance(model,i):
            if i in model.Omega:
                return model.alpha_g[i] + model.mu_max[i] - model.mu_min[i] - model.price == 0
            else:
                return model.alphas[i] + model.mu_max[i] - model.mu_min[i] - model.price == 0
        self.model.constraint_price_balance = Constraint(self.model.I, rule=rule_price_balance)

        self.model.constraint_power_balance = Constraint(rule=(self.demand)-sum(self.model.Pg[i] for i in self.model.I)==0) #type: ignore
        
        # Alphas inequalities
        def rule_inequality_up(model, i, j):
            return model.alpha_g[i] <= (model.aplhas[j] - self.tau) + self.bigM * (model.y_diff[j]) #type: ignore
        
        def rule_pmin(model,i):
            return model.Pg[i] <= self.bigM*model.zmin[i]
        
        self.model.constraint_pmin_binary = Constraint(self.model.I, rule= rule_pmin)

        def rule_mu_min_constraint(model,i):
            return model.mu_min[i]<= self.bigM*(1-model.zmin[i])
        
        self.model.constraint_mu_min_binary = Constraint(self.model.I, rule= rule_mu_min_constraint)

