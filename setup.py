from setuptools import setup, find_packages

setup(
    name="AML_WGAN",  # Nom du package
    version="0.1.0",  # Version du package
    description="AML fall course project at ENSAE (DSSA)",  # Description du projet
    url="https://github.com/NailKhelifa/AML_WGAN",  # URL du projet (par exemple sur GitHub)
    
    packages=find_packages(),  # Trouve automatiquement les packages dans ton projet
    install_requires=[  # Dépendances externes à installer
        "pandas", 
        "torch", 
        "numpy",
        "matplotlib", 
        "scikit-learn", 
        "h5py",
        "tqdm", 
        "seaborn", 
        "torchvision"
    ],
    
    python_requires='>=3.6',  # Version minimale de Python requise
)
