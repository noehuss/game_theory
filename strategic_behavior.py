# Strategic behavior for one producer
from pyomo.environ import * #type: ignore
import pandas as pd

class MPEC:
    def __init__(self, producer: str, alphas: list, marginal_costs: list, prod_df:pd.DataFrame, demand: int, bigM=1e3, tau=1.0, upper_bid=100, lower_bid=0):
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
        self.indexes()
        self.parameters()
        self.variables()
        self.constraints()
        self.objective_function()
        self.solve()

    def indexes(self):
        self.model.Omega = Set(initialize = self.prod_df[self.prod_df['producers']==self.producer].index.tolist())
        self.model.OmegaBar = Set(initialize = self.prod_df[self.prod_df['producers']!=self.producer].index.tolist())
        self.model.I = Set(initialize = self.prod_df.index.tolist())

    def parameters(self):
        self.model.capacities = Param(self.model.I, initialize = self.prod_df['capacities'].tolist())
        self.model.alphas = Param(self.model.I, initialize = self.alphas)

    def variables(self):
        # Production
        self.model.Pg = Var(self.model.I, domain=NonNegativeReals)

        # Bids
        self.model.alpha_g = Var(self.model.Omega, domain=NonNegativeReals, bounds=(self.lower_bid, self.upper_bid))
                
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

        self.model.constraint_power_balance = Constraint(rule=(self.demand-sum(self.model.Pg[i] for i in self.model.I)==0)) #type: ignore
        
        # Alphas inequalities
        def rule_inequality_up(model, i, j):
            return model.alpha_g[i] <= (model.alphas[j] - self.tau) + self.bigM * (model.y_diff[j]) #type: ignore
        self.model.constraint_in_up_alpha_g = Constraint(self.model.Omega, self.model.OmegaBar, rule=rule_inequality_up)

        def rule_inequality_down(model, i, j):
            return model.alpha_g[i] >= (model.alphas[j] + self.tau) - self.bigM * (1 - model.y_diff[j]) #type: ignore
        self.model.constraint_in_down_alpha_g = Constraint(self.model.Omega, self.model.OmegaBar, rule=rule_inequality_down)

        # Complementary slackness constraints
        def rule_capamax(model, i):
            return model.Pg[i] <= model.capacities[i]
        self.model.constraint_capamax = Constraint(self.model.I, rule=rule_capamax)

        def rule_pmin(model,i):
            return model.Pg[i] <= self.bigM*model.z_min[i]
        self.model.constraint_pmin_binary = Constraint(self.model.I, rule= rule_pmin)

        def rule_mu_min(model,i):
            return model.mu_min[i]<= self.bigM*(1-model.z_min[i])
        self.model.constraint_mu_min_binary = Constraint(self.model.I, rule= rule_mu_min)

        def rule_pgmax(model, i):
            return model.capacities[i] - model.Pg[i] <= self.bigM*model.z_max[i]
        self.model.constraint_pgmax = Constraint(self.model.I, rule=rule_pgmax)

        def rule_mu_max(model, i):
            return model.mu_max[i] <= self.bigM*(1-model.z_max[i])
        self.model.constraint_mu_max = Constraint(self.model.I, rule=rule_mu_max)

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
        self.model.y_diff.display()

    def get_profit(self) -> float:
        return - value(self.model.obj) # type: ignore
    
    def get_price(self):
        return value(self.model.price)
    
    def update_alphas(self):
        for i in self.prod_df[self.prod_df['producers']==self.producer].index.tolist():
            self.alphas[i] = value(self.model.alpha_g[i]) #type: ignore
        return self.alphas
    
    def get_results(self) -> pd.DataFrame:
        """Dataframe with production, bids and prices"""
        data = {
            'production': [value(self.model.Pg[i]) for i in self.model.I], #type: ignore
            'bids': self.update_alphas(),
            'producer': self.prod_df['producers'].to_list(),
            'capacities': [value(self.model.capacities[i]) for i in self.model.I] #type: ignore
        }

        return pd.DataFrame(data)

class MPEClinearized(MPEC):
    def objective_function(self):
        sum_cost = sum([self.marginal_costs[i]*self.model.Pg[i] for i in self.model.Omega]) #type: ignore
        sum_mu_max = sum([-self.model.mu_max[i] * self.model.capacities[i] for i in self.model.I]) #type: ignore
        sum_omega_bar = sum([self.model.alphas[i]*self.model.Pg[i] for i in self.model.OmegaBar]) #type: ignore
        sum_pmax = sum([self.model.mu_max[i]*self.model.capacities[i] for i in self.model.Omega]) #type: ignore

        self.model.obj = Objective(expr=(-(self.model.price*self.demand + sum_mu_max - sum_omega_bar + sum_pmax) + sum_cost), sense=minimize) #type: ignore
