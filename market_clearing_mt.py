# Market clearing algorithm
from pyomo.environ import * #type: ignore
import pandas as pd

class MarketClearingMT:
    def __init__(self, bids:list[list], marginal_costs:list, demand:list, prod_df:pd.DataFrame):
        """
        Market clearing algorithm.
        Bids: Bids of market players
        Marginal costs: True marginal costs of market players
        Demand: Electricity demand, MW
        Prod_df: Dataframe with producer characteristics
        """
        self.model = ConcreteModel()
        self.demand = demand
        self.bids = bids
        self.marginal_costs = marginal_costs
        self.prod_df = prod_df
        self.indexes()
        self.parameters()
        self.variables()
        self.constraints()
        self.objective_function()
        self.solve()

    def indexes(self):
        self.model.G = Set(initialize = self.prod_df.index.tolist())
        self.model.T = RangeSet(0, len(self.demand)-1) #0 to T-1

    def parameters(self):
        self.model.Demand = Param(self.model.T, initialize = self.demand)
        self.model.Pmax = Param(self.model.G, initialize = self.prod_df['Pmax'].tolist())
        self.model.Pmin = Param( self.model.G, initialize = self.prod_df['Pmin'].tolist())
        self.model.Ramp = Param( self.model.G, initialize = self.prod_df['ramp'].tolist())
        def bids_init(model, t, g):
            return self.bids[t][g]
        self.model.bids = Param(self.model.T, self.model.G, initialize = bids_init) # warn

    def variables(self):
        self.model.Pg = Var(self.model.T, self.model.G, domain=NonNegativeReals)

    def constraints(self):
        def rule_prod_max(model, t, g):
            return model.Pg[t, g] <= model.Pmax[g]
        self.model.prod_constraint_max = Constraint(self.model.T, self.model.G, rule = rule_prod_max)
        def rule_prod_min(model, t, g):
            return model.Pg[t, g] >= model.Pmin[g]
        self.model.prod_constraint_min = Constraint(self.model.T, self.model.G, rule = rule_prod_min )

        def rule_demand(model, t):
            return sum(model.Pg[t, g] for g in model.G) == model.Demand[t]
        self.model.eq_constraint = Constraint(self.model.T, rule = rule_demand)

        def rule_ramp_down(model, t, g):
            if t==0:
                return Constraint.Skip
            return model.Pg[t, g] >= model.Pg[t-1, g] - model.Ramp[g]
        self.model.ramp_down_constraint = Constraint(self.model.T, self.model.G, rule= rule_ramp_down)

        def rule_ramp_up(model, t, g):
            if t==0:
                return Constraint.Skip
            return model.Pg[t, g] <= model.Pg[t-1, g] + model.Ramp[g]
        self.model.ramp_up_constraint = Constraint(self.model.T, self.model.G, rule= rule_ramp_up)


    def objective_function(self):
        # objective function
        self.model.obj = Objective(expr = sum(sum(self.model.Pg[t, g]*self.model.bids[t, g] for g in self.model.G) for t in self.model.T), sense=minimize) # type: ignore

    def solve(self):
        # Dual
        self.model.dual = Suffix(direction=Suffix.IMPORT)
        # Create a solver
        solver = SolverFactory("gurobi", solver_io="python")  # Make sure Gurobi is installed and properly configured
        # Solve the model
        solution = solver.solve(self.model, tee=True)
        self.model.Pg.display()
        self.model.dual.display()

    def get_price(self, t) -> float:
        return value(self.model.dual[self.model.eq_constraint[t]]) #type: ignore
    
    def get_profits(self):
        return [value(self.model.Pg[t,g])*self.get_price(t) - value(self.model.Pg[t,g])*value(self.marginal_costs[g]) for g in self.model.G for t in self.model.T] # type: ignore

    def get_dispatch(self):
        return [[value(self.model.Pg[t,g]) for t in self.model.T] for g in self.model.G] #type: ignore
    
    def get_results(self, t) -> pd.DataFrame:
        """Dataframe with production, bids and prices"""
        data = {
            'production': [value(self.model.Pg[t,i]) for i in self.model.G], #type: ignore
            'bids': self.marginal_costs,
            'producer': self.prod_df['producers'].to_list(),
            'capacities': self.prod_df['capacities'].to_list() #type: ignore
        }
        print(data)
        return pd.DataFrame(data)
    
    def get_social_cost(self, t) -> float: #warn
        q = [value(self.model.Pg[t,g]) for g in self.model.G] #type: ignore
        return float(sum(float(self.marginal_costs[g]) * q[g] for g in self.model.G))
