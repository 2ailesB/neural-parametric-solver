from components.approximation.bases.fourier_basis import Fourier_basis
from components.approximation.bases.fourier_basis_adaptative import Fourier_basis_Adaptative
from components.approximation.bases.chebyshev_polynomials import Chebyshev_Poly
from components.approximation.bases.legendre_polynomials import Legendre_Poly
from components.approximation.bases.hermite_polynomials import Hermite_Poly
from components.approximation.bases.nn import hnet_basis
# from components.approximation.bases.polynomials import Polynomials
from components.approximation.bases.bsplines import Bsplines


approximations = {
    "fourier": Fourier_basis,
    "fourier_adapt": Fourier_basis_Adaptative,
    "chebyshev": Chebyshev_Poly,
    "legendre": Legendre_Poly,
    "hermite": Hermite_Poly,
    "hnet": hnet_basis,
    # "polynomial": Polynomials,
    "bsplines": Bsplines,
}

def init_1dbasis(name, cfg):
    try:
        return approximations[name](cfg)
    except KeyError : 
        raise NotImplementedError(
            f'Basis {name} not implemented')
