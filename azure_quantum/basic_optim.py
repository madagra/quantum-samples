# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from azure.quantum import Workspace
from azure.quantum.optimization import Problem, ProblemType, Term, ParallelTempering

workspace = Workspace (
  subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID"),
  resource_group = os.environ.get("AZURE_QUANTUM_RESOURCE_GROUP"),
  name = os.environ.get("AZURE_QUANTUM_WORKSPACE"),
  location = os.environ.get("AZURE_REGION", "westeurope")
)

# Define the problem
# problem types can be .ising or .pubo 
# for .ising the variables defining the cost function can take values [1, -1]
# for .pubo the values taken by the variables are instead [0, 1]
problem = Problem(name="My First Problem", problem_type=ProblemType.ising)

# this problem corresponds to a cost function of: 
# 2 x_0 -2 x_1 which is minimized by x_0 = -1 and x_1 = 1 with a final cost = -4 for .ising problem
terms = [
    Term(c=2, indices=[0]),
    Term(c=-2, indices=[1]),
]

problem.add_terms(terms=terms)

# Create the solver
solver = ParallelTempering(workspace, timeout=100)

# Solve the problem
result = solver.optimize(problem)
print(result)