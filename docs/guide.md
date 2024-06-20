# User Guide

Welcome to the user guide for `KitchenSink.jl`.

## Overview

This guide provides an overview of how to use the different modules and functions provided by `KitchenSink.jl`.

## Table of Contents

- [Core](@ref)
- [Meshes](@ref)
- [Error Estimation](@ref)
- [Problems](@ref)
- [Solvers](@ref)

## Core

### Abstract Types

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

### Concrete Types

```@docs
TensorProductMask
LocationMatrix
HierarchicalBasisFunction
DecayRate
```

## Meshes

### Types

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

## Error Estimation

### Types

```@docs
ResidualErrorEstimator
GoalOrientedErrorEstimator
```

## Problems

### Types

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

## Solvers

### Types

```@docs
Solver
TimeSteppingSolver
OptimalControlSolver
SolverLog
SolverMonitor
ParallelOptions
DerivativeRecovery
```
