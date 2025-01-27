from setuptools import find_packages, setup

setup(
    name="NGD",
    version="0.0.1",
    description="Package for Neural PDE Solver",
    author="Lise Le Boudec",
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
