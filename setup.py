import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nowcast_lstm",
    version="0.1.0",
    author="Daniel Hopp",
    author_email="daniel.hopp@un.org",
    description="Code for running LSTM neural networks on economic data for nowcasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhopp1/nowcast_lstm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
