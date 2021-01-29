import setuptools

with open('README.md', "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pylibsum",
    version="0.0.1",
    author="Tan Nian Wei",
    author_email="tannianwei@aggienetwork.com",
    description="Summarizes libraries used in a Python script/repo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnwei/pylibsum",
    package_dir={"": "src"},
    packages=["pylibsum"],
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={
        "console_scripts": ["pylibsum = pylibsum:main"]
    }
)
