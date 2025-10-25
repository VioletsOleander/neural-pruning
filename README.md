# Neural Pruning

A partial reproduction of the pruning techniques described in the paper [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) by Song Han et al.

This repository provides a pipeline to pre-train, prune, and test LeNet5 and LeNet300100 on the MNIST dataset.

## 1. Getting Started

### Prerequisites

This repository uses [Pixi](https://github.com/prefix-dev/pixi) for hassle-free dependency management. The target platform and device is linux-64 and cpu.

```bash
# Install pixi (if you haven't already)
curl -fsSL https://pixi.sh/install.sh | sh
```

### Setup

```bash
# Clone and enter the repository
git clone git@github.com:VioletsOleander/neural-pruning.git
cd neural-pruning
```

## 2. The Pruning Workflow

This repository utilizes pixi tasks to manage workflow. The basic syntax is:

```bash
pixi r <task-name> <task-arguments...>
```

The `train, test, prune` tasks pre-defined in this repository all accept only one argument, which indicates the path to the configuration file. If the path is a relative path, it is considered as relative to the project root directory.

This repository uses TOML config files. Example configurations are in `configs/examples/`.

### Step 0: Download the Dataset

```bash
# Download the MNIST dataset (saves to 'dataset/')
pixi r get-dataset
```

### Step 1: Pre-train a Network

Use the example configuration file to pre-train LeNet5:

```bash
pixi r train configs/examples/lenet5.example.toml
```

### Step 2: Prune the Pre-trained Network

Use the example configuration file to prune the pre-trained LeNet5:

```bash
pixi r prune configs/examples/lenet5.example.toml
```

### Step 3: Test the Model

Evaluate performance of the pruned model:

```bash
# Test the pruned model
pixi r test configs/examples/lenet5.example.toml
```

Then, update the `pruned` field under `[test]` section in the example configuration file `lenet5.example.toml`:

```toml
[test]
pruned = false
```

and evaluate the preformance of the original model:

```bash
# Test the pre-trained model 
pixi r test configs/examples/lenet5.example.toml
```

## 3. Results

### Model Comparison: LeNet5

Unpruned Model Checkpoint: `LeNet5_epoch10.pt`

Pruned Model Checkpoint: `PrunedLeNet5_iter3.pt`

#### Parameter Count Comparison

| Layer        | Parameters (K) Before Pruning   | Parameters (K) After Pruning   | Difference   | Reduction %   |
|--------------|---------------------------------|--------------------------------|--------------|---------------|
| conv1.weight | 0.15                            | 0.08                           | 0.07         | 48.7%         |
| conv1.bias   | 0.01                            | 0.01                           | 0.00         | 0.0%          |
| conv2.weight | 2.40                            | 0.66                           | 1.74         | 72.5%         |
| conv2.bias   | 0.02                            | 0.02                           | 0.00         | 0.0%          |
| fc1.weight   | 48.00                           | 6.00                           | 42.00        | 87.5%         |
| fc1.bias     | 0.12                            | 0.12                           | 0.00         | 0.0%          |
| fc2.weight   | 10.08                           | 0.65                           | 9.44         | 93.6%         |
| fc2.bias     | 0.08                            | 0.08                           | 0.00         | 0.0%          |
| fc3.weight   | 0.84                            | 0.61                           | 0.23         | 27.1%         |
| fc3.bias     | 0.01                            | 0.01                           | 0.00         | 0.0%          |
| **TOTAL**    | **61.71**                       | **8.23**                       | **53.48**    | **86.7%**     |

#### Model Size Comparison

| Layer        | Size (KB) Before Pruning   | Size (KB) After Pruning   | Difference   | Reduction %   |
|--------------|----------------------------|---------------------------|--------------|---------------|
| conv1.weight | 0.60                       | 0.31                      | 0.29         | 48.7%         |
| conv1.bias   | 0.02                       | 0.02                      | 0.00         | 0.0%          |
| conv2.weight | 9.60                       | 2.64                      | 6.96         | 72.5%         |
| conv2.bias   | 0.06                       | 0.06                      | 0.00         | 0.0%          |
| fc1.weight   | 192.00                     | 24.00                     | 168.00       | 87.5%         |
| fc1.bias     | 0.48                       | 0.48                      | 0.00         | 0.0%          |
| fc2.weight   | 40.32                      | 2.58                      | 37.74        | 93.6%         |
| fc2.bias     | 0.34                       | 0.34                      | 0.00         | 0.0%          |
| fc3.weight   | 3.36                       | 2.45                      | 0.91         | 27.1%         |
| fc3.bias     | 0.04                       | 0.04                      | 0.00         | 0.0%          |
| **TOTAL**    | **246.82**                 | **32.92**                 | **213.91**   | **86.7%**     |

#### FLOPs Comparison

| Layer     | FLOPs (K) Before Pruning   | FLOPs (K) After Pruning   | Difference   | Reduction %   |
|-----------|----------------------------|---------------------------|--------------|---------------|
| conv1     | 239.90                     | 125.44                    | 114.46       | 47.7%         |
| relu      | 0.08                       | 0.08                      | 0.00         | 0.0%          |
| pool1     | 4.70                       | 4.70                      | 0.00         | 0.0%          |
| conv2     | 481.60                     | 133.40                    | 348.20       | 72.3%         |
| pool2     | 1.60                       | 1.60                      | 0.00         | 0.0%          |
| fc1       | 96.12                      | 12.12                     | 84.00        | 87.4%         |
| fc2       | 20.24                      | 1.37                      | 18.87        | 93.2%         |
| fc3       | 1.69                       | 1.23                      | 0.46         | 27.0%         |
| **TOTAL** | **845.95**                 | **279.96**                | **565.99**   | **66.9%**     |

#### Summary

| Model        | Test Accuracy (%)   |   Parameters (K) | Compression Ratio   |
|--------------|---------------------|------------------|---------------------|
| LeNet5       | 97.00%              |            61.71 | 1.00x               |
| PrunedLeNet5 | 96.40%              |             8.23 | 7.50x               |

### Model Comparison: LeNet300100

Unpruned Model Checkpoint: `LeNet300100_epoch10.pt`

Pruned Model Checkpoint: `PrunedLeNet300100_iter3.pt`

#### Parameter Count Comparison

| Layer      | Parameters (K) Before Pruning   | Parameters (K) After Pruning   | Difference   | Reduction %   |
|------------|---------------------------------|--------------------------------|--------------|---------------|
| fc1.weight | 235.20                          | 29.40                          | 205.80       | 87.5%         |
| fc1.bias   | 0.30                            | 0.30                           | 0.00         | 0.0%          |
| fc2.weight | 30.00                           | 1.92                           | 28.08        | 93.6%         |
| fc2.bias   | 0.10                            | 0.10                           | 0.00         | 0.0%          |
| fc3.weight | 1.00                            | 0.73                           | 0.27         | 27.1%         |
| fc3.bias   | 0.01                            | 0.01                           | 0.00         | 0.0%          |
| **TOTAL**  | **266.61**                      | **32.46**                      | **234.15**   | **87.8%**     |

#### Model Size Comparison

| Layer      | Size (KB) Before Pruning   | Size (KB) After Pruning   | Difference   | Reduction %   |
|------------|----------------------------|---------------------------|--------------|---------------|
| fc1.weight | 940.80                     | 117.60                    | 823.20       | 87.5%         |
| fc1.bias   | 1.20                       | 1.20                      | 0.00         | 0.0%          |
| fc2.weight | 120.00                     | 7.68                      | 112.32       | 93.6%         |
| fc2.bias   | 0.40                       | 0.40                      | 0.00         | 0.0%          |
| fc3.weight | 4.00                       | 2.92                      | 1.08         | 27.1%         |
| fc3.bias   | 0.04                       | 0.04                      | 0.00         | 0.0%          |
| **TOTAL**  | **1066.44**                | **129.84**                | **936.60**   | **87.8%**     |

#### FLOPs Comparison

| Layer     | FLOPs (K) Before Pruning   | FLOPs (K) After Pruning   | Difference   | Reduction %   |
|-----------|----------------------------|---------------------------|--------------|---------------|
| fc1       | 470.70                     | 59.10                     | 411.60       | 87.4%         |
| relu      | 0.10                       | 0.10                      | 0.00         | 0.0%          |
| fc2       | 60.10                      | 3.94                      | 56.16        | 93.4%         |
| fc3       | 2.01                       | 1.47                      | 0.54         | 27.0%         |
| **TOTAL** | **532.91**                 | **64.61**                 | **468.30**   | **87.9%**     |

#### Summary

| Model             | Test Accuracy (%)   |   Parameters (K) | Compression Ratio   |
|-------------------|---------------------|------------------|---------------------|
| LeNet300100       | 95.62%              |           266.61 | 1.00x               |
| PrunedLeNet300100 | 94.76%              |            32.46 | 8.21x               |
