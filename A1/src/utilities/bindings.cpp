#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <random>

#include "Tensor.hpp"
#include "../layers/MaxPool.hpp"

namespace py = pybind11;

PYBIND11_MODULE(my_backend, m) {
    m.doc() = "GNR638 C++ Backend Plugin";

    
    py::class_<Tensor<float>>(m, "Tensor")
        
        .def(py::init([](const std::vector<int>& shape) {
            return new Tensor<float>(shape.size(), shape);
        }))
        .def(py::init([](const std::vector<float>& input_data, const std::vector<int>& target_shape) {
            int total_size = input_data.size();
            std::vector<int> dims_1d = {total_size};
            auto* t = new Tensor<float>(1, dims_1d);
            for (size_t i = 0; i < input_data.size(); ++i) {
                t->set(i, input_data[i]);
            }
            t->view(target_shape.size(), target_shape);
            return t;
        }))
        
        // math
        .def("matmul", &Tensor<float>::matmul)
        .def("transpose", &Tensor<float>::transpose)
        .def("convolve2D", &Tensor<float>::convolve2D)
        .def("zero", &Tensor<float>::zero)
        .def("sum", &Tensor<float>::sum)
        .def("col_sum", &Tensor<float>::col_sum)
        .def("print", &Tensor<float>::print)

        // Activation Functions
        .def("relu", &Tensor<float>::relu)
        .def("sigmoid", &Tensor<float>::sigmoid)
        .def("softmax", &Tensor<float>::softmax)
        .def("relu_derivative", &Tensor<float>::relu_derivative)
        .def("sigmoid_derivative", &Tensor<float>::sigmoid_derivative)

        .def("view", [](Tensor<float>& t, std::vector<int> new_dims) {
            t.view(new_dims.size(), new_dims);
        })

        .def("randn", [](Tensor<float>& t, int seed, double mean, double stddev, double multiplier) {
            std::default_random_engine generator(seed);
            std::normal_distribution<double> distribution(mean, stddev);
            t.randn(generator, distribution, multiplier);
        }, py::arg("seed")=42, py::arg("mean")=0.0, py::arg("stddev")=1.0, py::arg("multiplier")=1.0)

        .def("dropout", [](Tensor<float>& t, int seed, double p) {
            std::default_random_engine generator(seed);
            std::uniform_real_distribution<> distribution(0.0, 1.0);
            t.dropout(generator, distribution, p);
        }, py::arg("seed")=42, py::arg("p")=0.5)


        // Operators
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def(py::self * float())     
        .def(float() * py::self)     
        .def(py::self / float())
        .def(py::self -= py::self)

        // Getters 
        .def("get", (float (Tensor<float>::*)(int) const) &Tensor<float>::get)
        .def("get", (float (Tensor<float>::*)(int, int) const) &Tensor<float>::get)
        .def("get", (float (Tensor<float>::*)(int, int, int) const) &Tensor<float>::get)
        .def("get", (float (Tensor<float>::*)(int, int, int, int) const) &Tensor<float>::get)
        
        // Setters
        .def("set", (void (Tensor<float>::*)(int, float)) &Tensor<float>::set)
        .def("set", (void (Tensor<float>::*)(int, int, float)) &Tensor<float>::set)
        .def("set", (void (Tensor<float>::*)(int, int, int, float)) &Tensor<float>::set)
        .def("set", (void (Tensor<float>::*)(int, int, int, int, float)) &Tensor<float>::set)
        
        // Adders
        .def("add", (void (Tensor<float>::*)(int, float)) &Tensor<float>::add)
        .def("add", (void (Tensor<float>::*)(int, int, int, int, float)) &Tensor<float>::add)
        
        .def_property_readonly("shape", [](const Tensor<float>& t) {
            return t.dims; \
        });


    py::class_<MaxPool>(m, "MaxPool")
        .def(py::init<int, int>(), py::arg("size"), py::arg("stride"))
        .def("forward", &MaxPool::forward)
        .def("backprop", &MaxPool::backprop);
}