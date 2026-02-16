from setuptools import setup, Extension
import pybind11
import sys

# Define the C++ extension
cpp_args = ['-std=c++11', '-O3', '-DNDEBUG', '-march=native']

ext_modules = [
    Extension(
        'my_backend',
        sources=[
            'src/utilities/bindings.cpp',
            'src/utilities/Tensor.cpp',
            'src/layers/NetworkModel.cpp',
            'src/layers/LinearLRScheduler.cpp',
            'src/layers/ReLU.cpp',
            'src/layers/Sigmoid.cpp',
            'src/layers/SoftmaxClassifier.cpp',
            'src/layers/MaxPool.cpp',
            'src/layers/FullyConnected.cpp', 
            'src/layers/conv2d.cpp',
        ],
        include_dirs=[
            pybind11.get_include(),
            'src/utilities',
            'src/layers'
        ],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='my_backend',
    version='0.1',
    author='Your Name',
    description='C++ Backend for Neural Network',
    ext_modules=ext_modules,
)