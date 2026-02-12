#pragma once

#include <vector>
#include <random>
#include <cassert>

template<typename T>
class Tensor{
private:
    std::vector<T> data;
public:
    int num_dims = 0;
    int dims[4] = {0, 0, 0, 0};

    Tensor() = default;
    Tensor(int num_dims, int const* dims);
    Tensor(const Tensor<T> &other);
    void view(int new_num_dims, int* new_dims);
    void zero();

    T get(int i) const; // 1d tensor
    T get(int i, int j) const; // 2d tensor
    T get(int i, int j, int k) const; // 3d tensor
    T get(int i, int j, int k, int l) const; // 4d tensor

    void set(int i, T value); // 1d tensor
    void set(int i, int j, T value); // 2d tensor
    void set(int i, int j, int k, T value); // 3d tensor
    void set(int i, int j, int k, int l, T value); // 4d tensor
    void add(int i, T value);
    void add(int i, int j, int k, int l, T value);

    Tensor<T> matmul(Tensor<T> &other);
    Tensor<T> transpose();
    Tensor<T> relu();
    Tensor<T> softmax();
    Tensor<T> sigmoid();

    Tensor<T> convolve2D(Tensor<T> &kernel, int stride=1, int padding=0, Tensor<T> bias = Tensor<T>());
    void dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p);

    Tensor<T> sigmoid_derivative();
    Tensor<T> relu_derivative();

    T sum() const; // sum of all elements

    Tensor<T> operator+(Tensor<T> &other); // element-wise addition
    Tensor<T> operator*(Tensor<T> &other); // element-wise multiplication
    Tensor<T> operator*(T scalar); // scalar multiplication
    Tensor<T> operator/(T scalar); // scalar division

    Tensor<T>& operator-=(const Tensor<T>&difference);

    Tensor<T> col_sum() const;
    Tensor<T> channel_sum() const;
    void randn(std::default_random_engine generator, std::normal_distribution<double> distribution, double multiplier);

    void print() const; // print tensor contents
};