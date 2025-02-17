from setuptools import setup, find_packages

setup(
    name="pyshred",
    version="0.1.0",
    author="Kutz Research Group",
    author_email="pyshred1@gmail.com",
    description="PySHRED: A Python Package for SHallow REcurrent Decoders (SHRED) for Spatial-Temporal Systems",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PyShred-Dev/PyShred",
    packages=find_packages(include=["pyshred", "pyshred.*"]),
    license = "MIT",
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scikit-learn",
        "tqdm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True
)
