# Learning a neural solver for Parametric PDE to Enhance Physics-Informed Methods
Code for "Learning a neural solver for Parametric PDE to Enhance Physics-Informed Methods", poster at ICLR 2025. 

📋 [ICLR 2025](https://iclr.cc/virtual/2025/poster/28615)  
📑 [ArXiv](https://arxiv.org/abs/2410.06820)  
🤗 [Hugging Face](https://huggingface.co/datasets/2ailesB/neural-parametric-solver-datasets)  

## ...Progression...
This git will be progressively updated in the upcoming weeks with all experiments and cfg used.
Currently in progress. 


## Setup
```
conda create -n neural-parametric-solver python=3.10.11
pip install -e .
```

## Code overview
The main file to train a neural solver is `main.py`. 
To train a model, 3 python objects are required: a Dataset (folder Dataset), a Model (folder models) and a Trainer (folder training). 
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
We succintly detail the avalaible dataset in the paper. Datasets are provided on [Hugging Face](https://huggingface.co/datasets/2ailesB/neural-parametric-solver-datasets). For more details on the PDEs and the detailed setting, we refer to section C in the appendices of [our paper](https://openreview.net/pdf?id=jqVj8vCQsT).

- `helmholtz`: solve the 1d helmholtz equation $\frac{\partial^2 u (x)}{\partial x^2} + \omega^2u(x) = 0, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$ with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\omega \sim \mathcal{U}(0.5, 10)$

- `poisson`:  solve the 1d Poisson equation with constant forcing term : $\frac{\partial^2 u (x)}{\partial x^2} = \alpha, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$, with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\alpha \sim \mathcal{U}(0.5, 10)$. 

- `helmholtz-hf`: solve the 1d helmholtz equation $\frac{\partial^2 u (x)}{\partial x^2} + \omega^2u(x) = 0, u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$ with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $\omega \sim \mathcal{U}(0.5, 50)$

- `forcingmspoisson`: solve the 1d Poisson equation with forcing term : $\frac{\partial^2 u (x)}{\partial x^2} = f(x), u(0) = u_0, \frac{\partial u(0)}{\partial x} = v_0$, with $u_0\sim \mathcal{N}(0, 1)$, $v_0 \sim \mathcal{N}(0, 1)$ and $f(x) = \frac{\pi}{K}\sum_{i=1}^{K}\alpha_i i^{2r}\sin(i\pi x), a_i \sim \mathcal{U}(-100, 100), K=16, r=-0.5$.

- `1dnlrd`: solves a non-linear Reaction-Diffusion PDE. 

$$
\begin{aligned}
    \frac{\partial u(t, x)}{\partial t} - \nu \frac{\partial^2u(t, x)}{\partial x^2} - \rho u(t, x)(1-u(t, x)) &= 0, \\
    u(0, x) &= e^{-32(x-1/2)^2}.
\end{aligned}
$$

We generate $800$ trajectories by varying $\nu$ in $[1, 5]$ and $\rho$ in $[-5, 5]$.

- `1dnlrdics`: solves a non-linear Reaction-Diffusion PDE (see above), but the initial condition also varies as:

$$
\begin{equation}
    u(x, 0) = \sum_{i=1}^3 a_i e^{-\frac{\left( \frac{x-h/4}{h}\right)^2}{4}}.
\end{equation}
$$

Where $a_i$ are randomly chosen in $[0, 1]$ and $h=1$ is the spatial resolution. 

- `advection`: Take one dataset with fixed advection parameter from PDEBench [1]. We refer to PDEBench for more deatils. The PDE expresses as: 

$$
\begin{aligned}
    \frac{\partial u(t, x)}{\partial t} + \beta \frac{\partial u(t, x)}{\partial x} &= 0, \hspace{3mm} x\in (0, 1), t\in (0, 2], \\
    u(0, x) &= u_0(x), \hspace{3mm} x\in (0, 1).
\end{aligned}
$$

- `advections`: Adapted from PDEBench [1]: mix trajectories with several PDE parameters ($\beta$ varying between $0.2$ and $4$). 

- `darcy`: The Darcy dataset is talen from [2]. As for the `advection` dataset, we refer to the FNO paper for more details on the dataset. The PDE expresses as: 

$$
\begin{aligned}
    -\nabla.(a(x)\nabla u(x)) &= f(x) \hspace{3mm} x \in (0, 1)^2,\\
    u(x) &= 0 \hspace{3mm} x \in \partial(0, 1)^2.
\end{aligned}
$$

- `heat2d`: This dataset is inspired from [3]. The PDE expresses as: 
$$
\begin{aligned}
    \frac{\partial u (x, y, t)}{\partial t} - \nu \nabla^2 u(x, y, t) &= 0,\\
    u(x, y, 0) &= \sum_{j=1}^J A_j\sin(\frac{2\pi l_{xj}x}{L} + \frac{2\pi l_{yj}y}{L} + \phi_i).
\end{aligned}
$$


## Models

- `gd_nn`: non linear preconditioning of the gradient : $\theta_{k+1} = \theta_k - \eta P_{\theta}(\nabla \mathcal{L}(\theta), ...)$

## Training

- `nd`: General training training procedure for our model.


### References

[1] PDEBENCH: An Extensive Benchmark for Scientific Machine Learning, Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflüger, Mathias Niepert, NeurIPS 2022 - Track on Datasets and Benchmarks.

[2] Fourier Neural Operator for Parametric Partial Differential Equations, Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar, ICLR 2021. 

[3] Masked Autoencoders are PDE Learners, Anthony Zhou, Amir Barati Farimani, TMLR 20224. 