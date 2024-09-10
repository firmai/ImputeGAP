import pathlib
import setuptools

setuptools.setup(
    name="imputegap",
    version="0.1.0",
    description="Imputation tool for Time Series",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://exascale.info/",
    author="Quentin Nater",
    author_email="quentin.nater@unifr.ch",
    license="The Unlicense",
    project_urls = {
        "Documentation": "https://exascale.info/",
        "Source" : "https://exascale.info/"
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Topic :: Imputation"
    ],
    python_requires=">= 3.8,<3.10",
    install_requires=["pandas, numpy"],
    packages=setuptools.find_packages(),
    include_package_data=True
)