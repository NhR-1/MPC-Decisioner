from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="MPC-Decisioner",
    version="0.1.0",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    license="MIT",
    include_package_data=True,
)
