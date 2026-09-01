from setuptools import find_namespace_packages, setup

setup(
    name="m3drefclip",
    version="1.0",
    author="Yiming Zhang",
    description="M3DRef-CLIP Jetson benchmark",
    packages=find_namespace_packages(include=("m3drefclip", "m3drefclip.*")),
    install_requires=[
        f"clip @ git+https://github.com/eamonn-zh/CLIP.git", "lightning", "wandb", "scipy", "hydra-core",
        "h5py", "open3d", "pandas"
    ]
)
