from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.15.0", "scipy>=1.0.0",
                "joblib>=1.0.1", "scikit-learn>=0.22",
                "nilearn","sanssouci","matplotlib",
                "pandas", "joblib", "tqdm"]

setup(
    name="notip",
    version="0.1.2",
    author="Alexandre Blain",
    author_email="alexandre.blain@inria.fr",
    description="Nonparametric True Discovery Proportion control for brain imaging",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/alexblnn/Notip",
    download_url="https://github.com/alexblnn/Notip/releases/download/Notip_OHBM_v2/notip-0.1.2.tar.gz",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
