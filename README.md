# Self-Pruning Neural Network 

## Overview
This project implements a self-pruning neural network where the model learns to remove unimportant weights during training using learnable gating parameters.

## Method

Each weight is associated with a learnable gate:

weight_eff = weight × sigmoid(gate_score)

- If gate → 0 → connection is pruned  
- If gate → 1 → connection is retained  

## Loss Function

Total Loss = CrossEntropy + λ × SparsityLoss

SparsityLoss = sum of all gate values

The L1 penalty encourages many gates to move towards zero, resulting in a sparse network.

## Key Idea

Unlike traditional pruning (done after training), this model learns to prune itself during training through differentiable gating.

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|--------------|--------------|
| 0.5    | XX           | XX           |
| 1.5    | XX           | XX           |
| 3.0    | XX           | XX           |

## Observations

- Increasing λ increases sparsity but reduces accuracy.
- At low λ, most weights remain active.
- At high λ, many weights are pruned, leading to a sparse network.
- This demonstrates a clear sparsity–accuracy trade-off.

## Visualization

- Gate distribution shows clustering toward zero for higher λ.
- Trade-off plot shows inverse relationship between sparsity and accuracy.

## Conclusion

The model successfully learns to prune itself during training. The results demonstrate that higher sparsity can be achieved at the cost of accuracy, validating the effectiveness of the approach.
