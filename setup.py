import pathlib

import setuptools

setuptools.setup(
    name="imputegap",
    version="0.1.1",
    description="A Library of Imputation Techniques for Time Series Data",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eXascaleInfolab/ImputeGAP",
    author="Quentin Nater",
    author_email="quentin.nater@unifr.ch",
    license="The Unlicense",
    project_urls = {
        "Documentation": "https://github.com/eXascaleInfolab/ImputeGAP/tree/main",
        "Source" : "https://github.com/eXascaleInfolab/ImputeGAP"
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering"
    ],
    python_requires=">= 3.12.0,<3.12.8",
    install_requires=open('requirements.txt').read().splitlines(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
            'imputegap': ['env/*.toml', 'params/*.toml', 'dataset/*.txt'],  # Include the TOML files
    },
    entry_points={"console_scripts": ["imputegap = imputegap.runner_display:display_title"]}
)