#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Tensor.hpp"
#include "../layers/Module.hpp"
#include "../layers/ReLU.hpp"
#include "../layers/Sigmoid.hpp"
#include "../layers/MaxPool.hpp"
#include "../layers/SoftmaxClassifier.hpp"
#include "../layers/LinearLRScheduler.hpp"
#include "../layers/FullyConnected.hpp"
#include "../layers/conv2d.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_backend, m) {
    
    // 1. TENSOR
    // 1. TENSOR
    py::class_<Tensor<double>>(m, "Tensor")
        .def(py::init([](const std::vector<int>& shape) {
             return new Tensor<double>(shape.size(), shape);
        }))
        .def(py::init([](const std::vector<double>& data, const std::vector<int>& shape) {
             auto* t = new Tensor<double>(shape.size(), shape);
             // Fixed the loop comparison warning from your logs too
             for(size_t i=0; i < data.size(); ++i) t->set(i, data[i]);
             return t;
        }))
        .def(py::init<int, std::vector<int>>())
        .def("get", (double (Tensor<double>::*)(int) const) &Tensor<double>::get)
        .def("set", (void (Tensor<double>::*)(int, double)) &Tensor<double>::set)
        .def("matmul", &Tensor<double>::matmul)
        .def_property_readonly("shape", [](const Tensor<double>& t){ return t.dims; }) // <-- NO SEMICOLON HERE
        .def("to_list", &Tensor<double>::get_data)     // <-- Now this connects correctly
        .def("convolve2D", &Tensor<double>::convolve2D); // <-- SEMICOLON ONLY AT THE VERY END
    // 2. MODULE (Base Class)
    py::class_<Module>(m, "Module");

    // 3. LAYERS
    // IMPORTANT: use return_value_policy::reference because forward returns Tensor&
    
    py::class_<ReLU, Module>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward, py::return_value_policy::reference)
        .def("backprop", &ReLU::backprop);

    py::class_<Sigmoid, Module>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward, py::return_value_policy::reference)
        .def("backprop", &Sigmoid::backprop);

    py::class_<MaxPool, Module>(m, "MaxPool")
        .def(py::init<int, int>(), py::arg("size"), py::arg("stride"))
        .def("forward", &MaxPool::forward, py::return_value_policy::reference)
        .def("backprop", &MaxPool::backprop);

    py::class_<OutputLayer>(m, "OutputLayer");

    py::class_<SoftmaxClassifier, OutputLayer>(m, "SoftmaxClassifier")
        .def(py::init<>())
        .def("predict", &SoftmaxClassifier::predict) // Usually returns value
        .def("backprop", &SoftmaxClassifier::backprop);
        
    py::class_<LRScheduler>(m, "LRScheduler");
    
    py::class_<LinearLRScheduler, LRScheduler>(m, "LinearLRScheduler")
        .def(py::init<double, double>())
        .def_readwrite("learning_rate", &LinearLRScheduler::learning_rate);

    py::class_<FullyConnected, Module>(m, "FullyConnected")
        // Bind constructor with default seed = 0
        .def(py::init<int, int, int>(), py::arg("input_size"), py::arg("output_size"), py::arg("seed") = 0)
        // Forward returns reference
        .def("forward", &FullyConnected::forward, py::return_value_policy::reference)
        .def("backprop", &FullyConnected::backprop);

    py::class_<Conv2d, Module>(m, "Conv2d")
        .def(py::init<int, int, int, int, int, int>(), 
             py::arg("in_channels"), 
             py::arg("out_channels"), 
             py::arg("kernel_size"), 
             py::arg("stride"), 
             py::arg("padding"), 
             py::arg("seed") = 0)
        .def("forward", &Conv2d::forward, py::return_value_policy::reference)
        .def("backprop", &Conv2d::backprop);

    
    
    
}