from setuptools import find_packages, setup

setup(
    name="NGD",
    version="0.0.1",
    description="Package for Learning a Neural Solver for Parametric PDE to Enhance Physics-Informed Methods",
    author="Lise Le Boudec, Emmanuel de Bezenac, Louis Serrano, Ramon Daniel Regueiro-Espino, Yuan Yin, Patrick Gallinari",
    author_email="lise.leboudec@isir.upmc.fr",
    install_requires=["wandb",
        "torch>=2.0.0",
        "einops",
        "hydra-core",
        "wandb==0.14.0",
        "matplotlib",
        "numpy",
        "scipy",
    ],
    package_dir={"NGD": "NGD"},
    packages=find_packages(),
)

