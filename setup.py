from setuptools import setup, find_packages

setup(
    name="personalized_hrf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "tvb": ["tvb-framework", "tvb-library"],
        "dev": ["pytest", "sphinx", "black", "flake8"],
    },
    description="Personalized hemodynamic response function in computational models of fMRI activity",
    author="Demir Ege Ortaç",
    author_email="demiregeortac@gmail.com",
    license="MIT",
) 