from setuptools import setup, Extension
import pybind11
import os

# Define the C++ source files
# We list them explicitly to ensure the compiler finds them
cpp_sources = [
    # Utilities
    "src/utilities/bindings.cpp",
    "src/utilities/Tensor.cpp",
    
    # Layers (Current)
    "src/layers/ReLU.cpp",
    "src/layers/Sigmoid.cpp",
    "src/layers/MaxPool.cpp",
    "src/layers/SoftmaxClassifier.cpp",
    "src/layers/LinearLRScheduler.cpp",
    "src/layers/NetworkModel.cpp",
    
    # Layers (Future - Uncomment when you create these files)
    # "src/layers/FullyConnected.cpp",
    # "src/layers/Conv2D.cpp",
]

ext_modules = [
    Extension(
        "my_backend",
        sources=cpp_sources,
        include_dirs=[
            pybind11.get_include(),
            "src/utilities", # To find Tensor.hpp
            "src/layers"     # To find Layer headers
        ],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"], # -O3 for optimization
    ),
]

setup(
    name="my_backend",
    version="0.1",
    author="Your Name",
    description="A C++ Neural Network Backend",
    ext_modules=ext_modules,
)