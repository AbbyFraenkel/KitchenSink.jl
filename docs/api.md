# API Reference

## Modules

### Types

#### Core Types

```@docs
AbstractSystem
AbstractMesh
AbstractProblem
AbstractElement
AbstractTreeNode
AbstractConstraint
AbstractBoundaryCondition
AbstractObjectiveFunction
AbstractCostFunction
AbstractTensorProductMask
AbstractLocationMatrix
AbstractRefinementStrategy
AbstractErrorEstimator
AbstractIntegrationKernel
AbstractBasisFunction
AbstractHeiricialBasisFunction
```

#### Mesh Types

```@docs
Connectivity
Node
BasisFunction
Element
HierarchicalElement
Mesh
HierarchicalGridLayer
HRefinement
PRefinement
HPRefinement
```

#### Problem Types

```@docs
Equation
Domain
DirichletCondition
NeumannCondition
RobinCondition
BoundaryCondition
Weights
TimeSpan
DifferentialVars
```

#### Solver Types

```@docs
Solver
TimeSteppingSolver
OptimalControlSolver
SolverLog
SolverMonitor
ParallelOptions
DerivativeRecovery
```

#### Error Estimation Types

```@docs
ResidualErrorEstimator
GoalOrientedErrorEstimator
```
