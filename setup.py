from setuptools import setup, find_packages

setup(
    name="6g_marl_ran",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "pettingzoo",
        "numpy",
        "torch",
        "pandas",
        "matplotlib",
        "seaborn"
    ],
)
