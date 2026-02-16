#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <string>
#include <iostream>
#include <random>

template<typename T>
class Tensor {
public:
    std::vector<T> data; 
    std::vector<int> dims;
    int num_dims;

    // Constructors
    Tensor() : num_dims(0) {}
    Tensor(int num_dims, std::vector<int> dims);
    Tensor(const Tensor<T> &other);

    // Utilities
    void zero();
    void view(int new_num_dims, std::vector<int> new_dims);
    void print() const;
    std::vector<T> get_data() const; 
    T sum() const;

    // Element Access
    T get(int i) const;
    T get(int i, int j) const;
    T get(int i, int j, int k) const;
    T get(int i, int j, int k, int l) const;

    void set(int i, T value);
    void set(int i, int j, T value);
    void set(int i, int j, int k, T value);
    void set(int i, int j, int k, int l, T value);

    // Gradient Accumulation Helpers
    void add(int i, T value);
    void add(int i, int j, int k, int l, T value);

    // Operations
    Tensor<T> matmul(const Tensor<T>& other);
    Tensor<T> transpose();
    Tensor<T> col_sum() const;
    
    // Convolutions 
    Tensor<T> convolve2D(Tensor<T> &kernels, int stride, int padding, Tensor<T> bias);

    // Activations
    Tensor<T> relu();
    Tensor<T> relu_derivative();
    Tensor<T> sigmoid();
    Tensor<T> sigmoid_derivative();
    Tensor<T> softmax();

    // Random
    void randn(std::default_random_engine generator, std::normal_distribution<double> distribution, double multiplier);
    void dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p);

    // Operators
    Tensor<T> operator+(const Tensor<T>& other) const;
    Tensor<T> operator*(const Tensor<T> &other) const; 
    Tensor<T> operator*(T scalar) const;
    Tensor<T> operator/(T scalar) const;
    Tensor<T>& operator-=(const Tensor<T> &other);
};

#endif 