# Market clearing algorithm
from pyomo.environ import * #type: ignore
import pandas as pd

class MarketClearing:
    def __init__(self, bids:list, marginal_costs:list, demand:int, prod_df:pd.DataFrame):
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

    def parameters(self):
        self.model.capacities = Param(self.model.G, initialize = self.prod_df['capacities'].tolist())
        self.model.bids = Param(self.model.G, initialize = self.bids)

    def variables(self):
        self.model.Pg = Var(self.model.G, domain=NonNegativeReals)

    def constraints(self):
        def rule_prod(model, g):
            return model.Pg[g] <= model.capacities[g]
        self.model.prod_constraint = Constraint(self.model.G, rule= rule_prod)

        self.model.eq_constraint = Constraint(rule = sum(self.model.Pg[g] for g in self.model.G) == self.demand) # type: ignore


    def objective_function(self):
        # objective function
        self.model.obj = Objective(expr = sum(self.model.Pg[g]*self.model.bids[g] for g in self.model.G), sense=minimize) # type: ignore

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
        return [value(self.model.Pg[g])*self.get_price() - value(self.model.Pg[g])*value(self.marginal_costs[g]) for g in self.model.G] # type: ignore

    def get_dispatch(self):
        return [value(self.model.Pg[g]) for g in self.model.G] #type: ignore
    
    def get_results(self) -> pd.DataFrame:
        """Dataframe with production, bids and prices"""
        data = {
            'production': [value(self.model.Pg[i]) for i in self.model.G], #type: ignore
            'bids': self.marginal_costs,
            'producer': self.prod_df['producers'].to_list(),
            'capacities': self.prod_df['capacities'].to_list() #type: ignore
        }
        return pd.DataFrame(data)
    
    def get_social_cost(self) -> float:
        q = [value(self.model.Pg[g]) for g in self.model.G] #type: ignore
        return float(sum(float(self.marginal_costs[g]) * q[g] for g in self.model.G))
