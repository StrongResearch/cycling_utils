from setuptools import find_packages, setup

setup(
    name="cycling_utils",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["lightning==2.1.0rc0"],
)
