import torch
import matplotlib.pyplot as plt
import time

from components.approximation.FunctionBasis import FunctionBasis
from utils.gradients import gradients


class Bsplines(FunctionBasis):
    """
    B-spline basis functions
    
    Attributes:
        degree (int): The degree of the B-spline basis.
        N (int): Number of basis functions.
        knots_type (str): Type of knot placement, either 'equispaced' or 'shifted'.
        open_knots (bool): Whether to use open knots for B-spline generation.
        bias (bool): Whether to include a bias term in the basis.
        knots (torch.Tensor): Computed knot vector.
        dim (int): Dimensionality of the basis functions (default: 1).
        channels (int): Number of output channels (default: 1).
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.degree = cfg.degree
        self.N = cfg.N
        self.knots_type = cfg.knots_type
        self.open_knots = (self.knots_type == 'equispaced')
        self.bias = cfg.add_bias

        self.knots = self.get_knots()

        self.dim=1
        self.channels=1

    def get_knots(self):
        """
        Computes the knot vector for the B-spline basis.

        Returns:
            torch.Tensor: The computed knot vector.
        
        Raises:
            NotImplementedError: If an unsupported knot type is provided.
        """
        n_diff_knots = self.N + self.degree + 1

        if self.open_knots:
            n_diff_knots -= 2*self.degree

        assert n_diff_knots > 1, f'{n_diff_knots} different knots needed in this configuration'

        if self.knots_type == 'equispaced':
            knots = torch.linspace(0, 1., n_diff_knots)
        elif self.knots_type == 'shifted':
            sep = 1 / n_diff_knots
            knots = torch.linspace(0 - self.degree * sep, 1 + self.degree * sep, n_diff_knots)
        else:
            NotImplementedError(f'knot type {self.knots_type} not implemented')

        knots_list = [knots[0].reshape(1)] * self.degree + [knots] + [knots[-1].reshape(1)] * self.degree
        if self.open_knots:
            knots = torch.cat(knots_list)

        return knots

    def get_theta_size(self):
        print("self.N, self.bias : ", self.N, self.bias)

        return self.N + self.bias
    
    def to(self, device):
        self.device=device
        self.knots = self.knots.to(self.device)
    
    @staticmethod
    def compute_bspline_basis(x, degree, knots): 
        n_knots = len(knots)
        bsize, n_x = x.shape
        Bs = []

        # Initialize first degree B-splines in a tensorized way : B, X, knots-1 
        B = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).float()

        # Compute higher-degree B-spline basis functions 
        for p in range(1, degree+1):
            first_term = torch.zeros((bsize, n_x, n_knots-1-p))
            second_term = torch.zeros((bsize, n_x, n_knots-1-p))
            
            # t[i+k]-t[i] : i=0 start at p and in the end t[n] - t[n-p] to not exceed the last knot
            delta_knot1 = (knots[p:-1] - knots[:-p-1]) 
            # t[i+k+1]-t[i+1]
            delta_knot2 = knots[p+1:] - knots[1:-p] 
            
            # Avoid division by zero
            valid1 = (delta_knot1 != 0)
            valid2 = (delta_knot2 != 0)

            # Compute the first term of the recursive formula
            num = x.unsqueeze(-1) - knots[:-p-1]
            Bikm1 = B[:, :, :- 1]
            first_term = num / delta_knot1 * Bikm1
            first_term = torch.where(valid1, first_term, 0)

            # Compute the second term of the recursive formula
            num = knots[p+1:] - x.unsqueeze(-1)
            Bip1km1 = B[:, :, 1:]
            second_term = num / delta_knot2 * Bip1km1 
            second_term = torch.where(valid2, second_term, 0)
            
            # Compute the B-spline basis function of degree p
            B = first_term + second_term

        return B
    
    @staticmethod
    def compute_dbspline_basis(x, degree, knots, derivative): 
        "return bspline + derivative up to order derivative"
        n_knots = len(knots)
        bsize, n_x = x.shape

        order = degree - derivative
        assert order >= 0, f'derivative higher than {degree} are 0 for B-Spline of degree {degree}'

        # Initialize first degree B-splines in a tensorized way : B, X, knots-1
        B = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).float()
        dB = B.unsqueeze(-1) # B, X, knots-1, D + 1

        Bs = [B]
        dBs = [dB]

        # Compute higher-degree B-spline basis functions
        for p in range(1, degree+1):
            
            # t[i+k]-t[i] : i=0 on commence à p et à la fin t[n] - t[n-p] pour pas dépasser + le dernier sert pas comme premier terme car on peut pas avoir le deuxième terme ? 
            delta_knot1 = (knots[p:-1] - knots[:-p-1]) 
            # t[i+k+1]-t[i+1]
            delta_knot2 = knots[p+1:] - knots[1:-p] 
            
            valid1 = (delta_knot1 != 0)
            valid2 = (delta_knot2 != 0)


            num = x.unsqueeze(-1) - knots[:-p-1]
            Bikm1 = B[:, :, :- 1]
            first_term = num / delta_knot1 * Bikm1
            first_term = torch.where(valid1, first_term, 0)

            dBikm1 = dB[:, :, :-1,:]
            dfirst_term = torch.einsum('N, BXND -> BXND', p / delta_knot1, dBikm1)
            dfirst_term = torch.where(valid1.unsqueeze(0).unsqueeze(0).unsqueeze(-1), dfirst_term, 0)

            num = knots[p+1:] - x.unsqueeze(-1)
            Bip1km1 = B[:, :, 1:]
            second_term = num / delta_knot2 * Bip1km1 
            second_term = torch.where(valid2, second_term, 0)

            dBip1km1 = dB[:, :, 1:, :]
            dsecond_term = torch.einsum('N, BXND -> BXND', - p / delta_knot2, dBip1km1)
            dsecond_term = torch.where(valid2.unsqueeze(0).unsqueeze(0).unsqueeze(-1), dsecond_term, 0)
            
            B = first_term + second_term
            dB = torch.cat((B.unsqueeze(-1), dfirst_term + dsecond_term), dim=-1)

            Bs += [B]
            dBs += [dB]

        out = dB[:, :, :, :(derivative+1)]

        return out
    

    def compute_u(self, x, theta):

        basis = self.compute_bspline_basis(x, self.degree, self.knots)
        bias = torch.zeros(1).to(self.device)

        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        out = torch.bmm(basis, theta) # 'BXN, BNC -> BXC'

        return out + self.bias * bias.unsqueeze(-1)
    
    def compute_uderivativex(self, x, theta):
        basis = self.compute_dbspline_basis(x, self.degree, self.knots, 1)[:, :, :, 1] # On recupère la dernière spline de degré degree et la première derivative
        bias = torch.zeros(1).to(self.device)
        
        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        out = torch.bmm(basis, theta) # 'BXN, BNC -> BXC'
        return out # no bias when derivative wrt x
    
    def compute_uderivativex2(self, x, theta):
        basis = self.compute_dbspline_basis(x, self.degree, self.knots, 2)[:, :, :, 2]
        bias = torch.zeros(1).to(self.device)

        if self.bias:
            bias = theta[:, -1]
            theta = theta[:, :-1]

        out = torch.bmm(basis, theta) # 'BXN, BNC -> BXC'
        return out # no bias # + self.bias * bias.unsqueeze(-1)

    def get_basis(self, x):
        return self.compute_bspline_basis(x, self.degree, self.knots)
    
    def get_basis_derivativex(self, x):
        return self.compute_dbspline_basis(x, self.degree, self.knots, 1)[:, :, :, 1]

    def get_basis_derivativex2(self, x):
        return self.compute_dbspline_basis(x, self.degree, self.knots, 2)[:, :, :, 2]
    
    

def compute_dbspline_basis(x, degree, knots, derivative): 
    n_knots = len(knots)
    bsize, n_x = x.shape
    # out_size = 

    order = degree - derivative
    assert order >= 0, f'derivative higher than {degree} are 0 for B-Spline of degree {degree}'

    # Initialize first degree B-splines in a tensorized way : B, X, knots-1
    B = ((x.unsqueeze(-1) >= knots[:-1]) & (x.unsqueeze(-1) < knots[1:])).float()
    dB = B.unsqueeze(-1) # B, X, knots-1, D + 1

    Bs = [B]
    dBs = [dB]

    # Compute higher-degree B-spline basis functions
    for p in range(1, degree+1):
        
        # t[i+k]-t[i] : i=0 on commence à p et à la fin t[n] - t[n-p] pour pas dépasser + le dernier sert pas comme premier terme car on peut pas avoir le deuxième terme ? 
        delta_knot1 = (knots[p:-1] - knots[:-p-1]) 
        # t[i+k+1]-t[i+1]
        delta_knot2 = knots[p+1:] - knots[1:-p] 
        
        valid1 = (delta_knot1 != 0)
        valid2 = (delta_knot2 != 0)

        num = x.unsqueeze(-1) - knots[:-p-1]
        Bikm1 = B[:, :, :- 1]
        first_term = num / delta_knot1 * Bikm1
        first_term = torch.where(valid1, first_term, 0)

        dBikm1 = dB[:, :, :-1,:]
        dfirst_term = torch.einsum('N, BXND -> BXND', p / delta_knot1, dBikm1)
        dfirst_term = torch.where(valid1.unsqueeze(0).unsqueeze(0).unsqueeze(-1), dfirst_term, 0)

        num = knots[p+1:] - x.unsqueeze(-1)
        Bip1km1 = B[:, :, 1:]
        second_term = num / delta_knot2 * Bip1km1 
        second_term = torch.where(valid2, second_term, 0)

        dBip1km1 = dB[:, :, 1:, :]
        dsecond_term = torch.einsum('N, BXND -> BXND', - p / delta_knot2, dBip1km1)
        dsecond_term = torch.where(valid2.unsqueeze(0).unsqueeze(0).unsqueeze(-1), dsecond_term, 0)
        
        B = first_term + second_term
        dB = torch.cat((B.unsqueeze(-1), dfirst_term + dsecond_term), dim=-1)

        Bs += [B]
        dBs += [dB]

    return dBs

    

if __name__ =='__main__':
    from omegaconf import DictConfig

    # Example usage
    derivative = 2
    x = torch.linspace(0, 1.0, 100)

    cfg = DictConfig({'degree': 3, 'N': 10, 'knots_type': 'shifted', 'open_knots': False, 'add_bias': False})
    # usefull : equispaced open
    # shifted close
    degree = cfg.degree
    knots_type = cfg.knots_type
    print("cfg : ", cfg)

    bs = Bsplines(cfg)
    B = bs.compute_bspline_basis(x.unsqueeze(0), cfg.degree, bs.knots)
    print("B.shape : ", B.shape)
    print("x.shape : ", x.shape)
    plt.plot(x, B[0, :, :])
    plt.vlines(bs.knots, 0, 1, ls='dashed', color='silver', label='knots with multiplicity 1')
    plt.title(f"B-spline Basis Functions")
    plt.xlabel("x")
    plt.ylabel("Basis Value")
    plt.legend()
    plt.savefig(f'xp/vis/bsplines_{knots_type}.png')
    plt.clf()

    cfg = DictConfig({'degree': 3, 'N': 10, 'knots_type': 'equispaced', 'open_knots': False, 'add_bias': False})
    knots_type = cfg.knots_type
    bs = Bsplines(cfg)
    B = bs.compute_bspline_basis(x.unsqueeze(0), cfg.degree, bs.knots)
    print("B.shape : ", B.shape)
    print("x.shape : ", x.shape)
    plt.plot(x, B[0, :, :])
    plt.vlines(bs.knots[1:-1], 0, 1, ls='dashed', color='silver', label='knots with multiplicity 1')
    plt.vlines(bs.knots[[0, -1]], 0, 1, ls='dashed', color='k', label=f'knots with multiplicity {degree}')
    plt.title(f"B-spline Basis Functions")
    plt.xlabel("x")
    plt.ylabel("Basis Value")
    plt.legend()
    plt.savefig(f'xp/vis/bsplines_{knots_type}.png')
    plt.clf()
    exit()



    B = bs.compute_dbspline_basis(x.unsqueeze(0), cfg.degree, bs.knots, derivative)
    print("B.shape : ", B.shape)
    
    B = compute_dbspline_basis(x.unsqueeze(0), cfg.degree, bs.knots, derivative)
    # Plotting the basis functions
    for deg in range(degree+1):
        print("B[deg].shape : ", B[deg].shape)
        fig,axs = plt.subplots(deg+1, sharex=True)#, figsize=(deg*7, 7))
        if deg == 0:
            plt.plot(x, B[deg][0, :, :, 0])
            plt.title(f"{deg} B-spline Basis Functions")
            plt.xlabel("x")
            plt.ylabel("Basis Value")
        else :
            for der in range(B[deg].shape[3]):
                axs[der].plot(x, B[deg][0, :, :, der])
                axs[der].set_title(f"{deg} B-spline Basis Functions, {der}-th derivative ")
        plt.savefig(f'xp/vis/bsplines_chatgpt_der_{deg}_eq11.png')
        plt.clf()

