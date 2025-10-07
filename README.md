# neural-solver-parametric
Code for "Learning a neural solver for Parametric PDE to Enhance Physics-Informed Methods", poster at ICLR 2025. 

## Setup
```
conda create -n neural-parametric-solver python=3.10.11
pip install -e .
```

## ...Progression...
This git will be progressively updated in the upcoming weeks with all experiments and cfg used.
Currently in progress. 
Some of the provided code has not yet been checked.

## Code overview
The main file to train a neural solver is `main.py`. 
To train a model, 3 python objects are required: a Dataset (folder Dataset), a Model (folder models) and a Training (folder training). 
Each of them is instanciated with the desired config with the coresponding `init_****` functions that are available in the `init` folder. 
The default config file is `config/base.yaml`. It **purposefully** **doesn't take any input** in the network. This has to be manually added by setting the desired `input_***` parameters to True (1). See example below. 
To train a neural solver with specific new configs, you can add them in the command. For example, to use another dataset and specify the learning rate, use : 
```
python3 main.py dataset=helmholtz exp.lr=0.01 model.input_bc=1 model.input_gradtheta=1
```
The available config are summarized in the `params.txt` file. 


## How to reproduce the results of the paper? 

### Table 2: Training your neural parametric solver
- `helmholtz` 
- `poisson` 
- `nlrd` 
- `darcy-flow` 
- `heat` 

### Ablations
#### Number of steps L (figure 11)

#### Training with differents dataset size (figure 12)

#### Optimizing with $\mathcal{L}_{PDE} (table 7)

#### Solver configutation (table 8)

#### Inner learning rate ablation (table 9)

#### Input features (table 10)

#### Non linear basis (table 11)

#### NN layer (table 12) 

#### Irregular grids (table 13)

#### Additional datasets (table 18)


## Datasets

- `helmholtz`: solve the 1d helmholtz equation $\frac{\partial^2 u (x)}{\partial x^2} + \omega^2u(x) = 0, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$ with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\omega \sim \mathcal{U}(0.5, 10)$

- `poisson`:  solve the 1d Poisson equation with constant forcing term : $\frac{\partial^2 u (x)}{\partial x^2} = \alpha, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$, with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\alpha \sim \mathcal{U}(0.5, 10)$. 

- `helmholtz-hf`: solve the 1d helmholtz equation $\frac{\partial^2 u (x)}{\partial x^2} + \omega^2u(x) = 0, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$ with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\omega \sim \mathcal{U}(0.5, 50)$

- `forcingmspoisson`: solve the 1d Poisson equation with forcing term : $\frac{\partial^2 u (x)}{\partial x^2} = f(x), u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$, with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $f(x) = \frac{\pi}{K}\sum_{i=1}^{K}\alpha_i i^{2r}\sin(i\pi x), a_i \sim \mathcal{U}(-100, 100), K=16, r=-0.5$.

- `1dnlrd`: 
- `1dnlrdics`:
- `advection`: 
- `advections`: 
- `darcy`: 
- `heat2d`: 

Datasets are provided on [Hugging Face](https://huggingface.co/datasets/2ailesB/neural-parametric-solver-datasets).


## Models

- `gd_nn`: non linear preconditioning of the gradient : $\theta_{k+1} = \theta_k - \eta P_{\theta}(\nabla \mathcal{L}(\theta), ...)$


## Training

- `nd`: 