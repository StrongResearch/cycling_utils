from setuptools import setup

setup(
    name="cycling_utils",
    version="0.0.1",
    packages=["cycling_utils"],
    install_requires=[],
    extras_require={"lightning": ["lightning==2.1.0rc0"]},
)
