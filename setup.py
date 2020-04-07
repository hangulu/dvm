import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setuptools.setup(
    name="dvm",
    version="1.0.0",
    author="Hakeem Angulu",
    author_email="hakeem.angulu@gmail.com",
    description="The Discrete Voter Model for ecological inference.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hangulu/dvm",
    packages=setuptools.find_packages(),
    keywords=["voting", "ecological inference", "gerrymandering"],
    install_requires=[
        "numpy",
        "pymc3",
        "pandas",
        "tqdm",
        "tensorflow-probability",
        "tensorflow",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
