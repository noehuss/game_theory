# Strategic behavior for one producer
from pyomo.environ import * #type: ignore
import pandas as pd

class MPEC:
    def __init__(self, producer: str, alphas: list[list], marginal_costs: list, prod_df:pd.DataFrame, demand: list, bigM=1e3, tau=1.0, upper_bid=100, lower_bid=0):
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
        self.T = len(demand)-1
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
        self.model.T = RangeSet(0, self.T) #0 to T-1

    def parameters(self):
        self.model.Demand = Param(self.model.T, initialize = self.demand)
        self.model.Pmax = Param(self.model.I, initialize = self.prod_df['Pmax'].tolist())
        self.model.Pmin = Param(self.model.I, initialize = self.prod_df['Pmin'].tolist())
        self.model.Ramp = Param( self.model.I, initialize = self.prod_df['ramp'].tolist())
        def bids_init(model, t, i):
            return self.alphas[t][i]
        self.model.alphas = Param(self.model.T, self.model.I, initialize = bids_init)


    def variables(self):
        # Production
        self.model.Pg = Var(self.model.T, self.model.I, domain=NonNegativeReals)

        # Bids
        self.model.alpha_g = Var(self.model.T, self.model.Omega, domain=NonNegativeReals, bounds=(self.lower_bid, self.upper_bid))
                
        self.model.price = Var(self.model.T, domain=Reals)

        self.model.mu_max = Var(self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.mu_min = Var(self.model.T, self.model.I, domain=NonNegativeReals)

        # Binary variables
        self.model.z_max = Var(self.model.T, self.model.I, domain = Binary)
        self.model.z_min = Var(self.model.T, self.model.I, domain = Binary)

        # Alphas
        self.model.y_diff = Var(self.model.T, self.model.OmegaBar, domain=Binary)

        # Ramp constraints
        self.model.theta_max = Var(self.model.T, self.model.I, domain=NonNegativeReals)
        self.model.theta_min = Var(self.model.T, self.model.I, domain=NonNegativeReals)

        self.model.x_max = Var(self.model.T, self.model.I, domain = Binary)
        self.model.x_min = Var(self.model.T, self.model.I, domain = Binary) 

    def constraints(self):
        # Price and power balance
        def rule_price_balance(model, t, i):
            if t == 0:
                 expr_theta = - model.theta_max[t+1, i] + model.theta_min[t+1, i]
            elif t == self.T:
                expr_theta = model.theta_max[t, i] - model.theta_min[t, i]
            else:
                expr_theta = model.theta_max[t, i] - model.theta_min[t, i] - model.theta_max[t+1, i] + model.theta_min[t+1, i]
            expr = model.mu_max[t, i] - model.mu_min[t, i] - model.price[t] + expr_theta
            if i in model.Omega:
                return model.alpha_g[t, i] + expr == 0
            else:
                return model.alphas[t, i] + expr == 0
        self.model.constraint_price_balance = Constraint(self.model.T, self.model.I, rule=rule_price_balance)

        def rule_eq_demand(model, t):
            return sum(model.Pg[t, i] for i in self.model.I)==model.Demand[t]

        self.model.constraint_power_balance = Constraint(self.model.T, rule=rule_eq_demand) #type: ignore
        
        # Alphas inequalities
        def rule_inequality_up(model, t, i, j):
            return model.alpha_g[t, i] <= (model.alphas[t, j] - self.tau) + self.bigM * (model.y_diff[t, j]) #type: ignore
        self.model.constraint_in_up_alpha_g = Constraint(self.model.T, self.model.Omega, self.model.OmegaBar, rule=rule_inequality_up)

        def rule_inequality_down(model, t, i, j):
            return model.alpha_g[t, i] >= (model.alphas[t, j] + self.tau) - self.bigM * (1 - model.y_diff[t, j]) #type: ignore
        self.model.constraint_in_down_alpha_g = Constraint(self.model.T, self.model.Omega, self.model.OmegaBar, rule=rule_inequality_down)

        # Complementary slackness constraints
        def rule_capamax(model, t, i):
            return model.Pg[t, i] <= model.Pmax[i]
        self.model.constraint_capamax = Constraint(self.model.T, self.model.I, rule=rule_capamax)

        def rule_capamin(model, t, i):
            return model.Pg[t, i] >= model.Pmin[i]
        self.model.constraint_capamin = Constraint(self.model.T, self.model.I, rule=rule_capamin)

        def rule_pmin(model, t, i):
            return model.Pg[t, i] - model.Pmin[i] <= self.bigM*model.z_min[t, i]
        self.model.constraint_pmin_binary = Constraint(self.model.T, self.model.I, rule= rule_pmin)

        def rule_mu_min(model, t, i):
            return model.mu_min[t, i]<= self.bigM*(1-model.z_min[t, i])
        self.model.constraint_mu_min_binary = Constraint(self.model.T, self.model.I, rule= rule_mu_min)

        def rule_pmax(model, t, i):
            return model.Pmax[i] - model.Pg[t, i] <= self.bigM*model.z_max[t, i]
        self.model.constraint_pgmax = Constraint(self.model.T, self.model.I, rule=rule_pmax)

        def rule_mu_max(model, t, i):
            return model.mu_max[t, i] <= self.bigM*(1-model.z_max[t, i])
        self.model.constraint_mu_max = Constraint(self.model.T, self.model.I, rule=rule_mu_max)

        def rule_ramp_down_min(model, t, i):
            if t == 0:
                return Constraint.Skip
            return model.Pg[t, i] - model.Pg[t-1, i] + model.Ramp[i] >= 0
        self.model.constraint_ramp_down_min = Constraint(self.model.T, self.model.I, rule =rule_ramp_down_min)

        def rule_ramp_down_max(model, t, i):
            if t == 0:
                return model.x_min[t,i] == 1
            return model.Pg[t, i] - model.Pg[t-1, i] + model.Ramp[i] <= self.bigM*model.x_min[t, i]
        self.model.constraint_ramp_down_max = Constraint(self.model.T, self.model.I, rule =rule_ramp_down_max)

        def rule_ramp_up_min(model, t, i):
            if t == 0:
                return Constraint.Skip
            return - model.Pg[t, i] + model.Pg[t-1, i] + model.Ramp[i] >= 0
        self.model.constraint_ramp_up_min = Constraint(self.model.T, self.model.I, rule=rule_ramp_up_min)
        
        def rule_ramp_up_max(model, t, i):
            if t==0:
                return model.x_max[t, i] == 1
            return - model.Pg[t, i] + model.Pg[t-1, i] + model.Ramp[i] <= self.bigM*model.x_max[t, i]
        self.model.constraint_ramp_up_max = Constraint(self.model.T, self.model.I, rule=rule_ramp_up_max)

        def rule_theta_min(model, t, i):
            return model.theta_min[t, i]<= self.bigM*(1-model.x_min[t, i])
        self.model.constraint_theta_min_binary = Constraint(self.model.T, self.model.I, rule= rule_theta_min)

        def rule_theta_max(model, t, i):
            return model.theta_max[t, i]<= self.bigM*(1-model.x_max[t, i])
        self.model.constraint_theta_max_binary = Constraint(self.model.T, self.model.I, rule= rule_theta_max)

    def objective_function(self):
        # objective function
        self.model.obj = Objective(expr=sum(sum([-self.model.price[t]*self.model.Pg[t, i] + self.marginal_costs[i]*self.model.Pg[t, i] for i in self.model.Omega]) for t in self.model.T), sense=minimize) #type: ignore

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
        self.model.theta_max.display()
        self.model.theta_min.display()

    def get_profit(self) -> float:
        return - value(self.model.obj) # type: ignore
    
    def get_price(self, t):
        return value(self.model.price[t])
    
    def update_alphas(self):
        alphas = self.alphas.copy()
        for t in self.model.T:
            for i in self.prod_df[self.prod_df['producers']==self.producer].index.tolist():
                alphas[t][i] = value(self.model.alpha_g[t, i]) #type: ignore
        return alphas
    
    def get_results(self, t) -> pd.DataFrame:
        """Dataframe with production, bids and prices"""
        data = {
            'production': [value(self.model.Pg[t, i]) for i in self.model.I], #type: ignore
            'bids': self.update_alphas()[t],
            'producer': self.prod_df['producers'].to_list(), #type: ignore
            'capacities': self.prod_df['capacities'].to_list(), #type: ignore
            'Pmax': [value(self.model.Pmax[i]) for i in self.model.I], #type: ignore
            'Pmin': [value(self.model.Pmin[i]) for i in self.model.I] #type: ignore
        }
        print(data)

        return pd.DataFrame(data)

class MPEClinearized(MPEC):
    def objective_function(self):
        sum_cost = sum(sum([self.marginal_costs[i]*self.model.Pg[t, i] for i in self.model.Omega]) for t in self.model.T) #type: ignore
        sum_mu = sum(sum([self.model.mu_min[t, i] * self.model.Pmin[i]-self.model.mu_max[t,i] * self.model.Pmax[i] for i in self.model.I]) for t in self.model.T) #type: ignore
        sum_omega_bar = sum(sum([self.model.alphas[t, i]*self.model.Pg[t, i] for i in self.model.OmegaBar]) for t in self.model.T) #type: ignore
        sum_pmax = sum(sum([self.model.mu_max[t, i]*self.model.Pmax[i] for i in self.model.Omega]) for t in self.model.T) #type: ignore
        sum_pmin = sum(sum([self.model.mu_min[t, i]*self.model.Pmin[i] for i in self.model.Omega]) for t in self.model.T) #type: ignore
        sum_gains = sum(self.model.price[t]*self.demand[t] for t in self.model.T)
        sum_thetas = sum(sum((self.model.theta_max[t, i]-self.model.theta_min[t, i])*self.model.Ramp[i] for i in self.model.OmegaBar) for t in self.model.T)
        self.model.obj = Objective(expr=(-(sum_gains + sum_mu - sum_omega_bar - sum_pmin+ sum_pmax + sum_thetas) + sum_cost), sense=minimize) #type: ignore
