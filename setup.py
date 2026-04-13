from setuptools import setup, find_packages

setup(
    name="llm_unlearn",
    version="0.1.0",
    packages=find_packages(),
    author="Davygupta47",
    author_email="dwaipayan.dg07@gmail.com",
    description="Unlearning in Pre-trained Language Models (PLMs).",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Davygupta47/unlearning-plm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)