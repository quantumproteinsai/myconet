from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "numpy>=1.24",
    "scipy>=1.10",
    "matplotlib>=3.7",
    "scikit-learn>=1.3",
    "POT>=0.9",
]

setup(
    name                          = "myconet",
    version                       = "1.0.0",
    author                        = "Bertrand Mercier des Rochettes",
    author_email                  = "contact@quantum-proteins.ai",
    description                   = (
        "Python simulation framework for mycorrhizal network biophysics "
        "with Freiman-Villani thermodynamic analysis"
    ),
    long_description              = long_description,
    long_description_content_type = "text/markdown",
    url                           = "https://github.com/quantumproteinsai/myconet",
    project_urls                  = {
        "Paper"         : "https://arxiv.org/abs/2026.XXXXX",
        "Theory paper"  : "https://arxiv.org/abs/2026.YYYYY",
        "Bug Tracker"   : "https://github.com/quantumproteinsai/myconet/issues",
    },
    packages                      = find_packages(exclude=["tests*", "examples*"]),
    python_requires               = ">=3.10",
    install_requires              = install_requires,
    extras_require                = {
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers                   = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    keywords = (
        "mycorrhizal networks biophysics simulation "
        "Freiman index optimal transport Wasserstein "
        "Fokker-Planck mathematical biology"
    ),
    entry_points = {
        "console_scripts": [
            "myconet-drought=examples.drought_stress:main",
        ],
    },
)
