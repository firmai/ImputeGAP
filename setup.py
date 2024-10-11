import pathlib

import setuptools

setuptools.setup(
    name="imputegap",
    version="0.1.0",
    description="A Library of Imputation Techniques for Time Series Data",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/eXascaleInfolab/ImputeGAP",
    author="Quentin Nater",
    author_email="quentin.nater@unifr.ch",
    license="The Unlicense",
    project_urls = {
        "Documentation": "https://github.com/eXascaleInfolab/ImputeGAP/tree/main/docs",
        "Source" : "https://github.com/eXascaleInfolab/ImputeGAP"
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Topic :: Imputation"
    ],
    python_requires=">= 3.12.0,<3.12.6",
    install_requires=open('requirements.txt').read().splitlines(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["imputegap = imputegap.runner_display:display_title"]}
)