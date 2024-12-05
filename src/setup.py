from setuptools import find_packages, setup

setup(
    name="luxai-s3",
    version="0.1.0",
    packages=find_packages(exclude="kits"),
    install_requires=[
        "jax",
        "gymnax==0.0.8",
        "tyro",
    ],
    entry_points={"console_scripts": ["luxai-s3 = luxai_runner.cli:main"]},
    author="Lux AI Challenge",
    description="Lux AI Challenge Season 3 environment code",
)
