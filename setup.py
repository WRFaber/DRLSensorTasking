from setuptools import find_packages, setup

install_requires = ["tensorflow"]

data_files = []

setup(
    name="DRL_SENSOR_TASKING",
    version="0.1.0",
    description="DQN for sensor tasking",
    author="Weston Faber",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require={"dev": ["black", "pylint", "setuptools", "wheel"]},
    data_files=data_files,
)
