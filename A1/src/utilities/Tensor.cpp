#include "Tensor.hpp"

template
class Tensor<int>;

template
class Tensor<float>;

template
class Tensor<double>;


template<typename T>
Tensor<T>::Tensor(int num_dims, int const* dims) {

    assert(num_dims > 0 && num_dims <= 4);
    int size = 1;
    for (int i=0; i<num_dims; i++)
    {
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    data.resize(size);
    this->num_dims = num_dims;
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) : num_dims(other.num_dims) {

    std::copy(other.dims, other.dims+4, this->dims);
    this->data = other.data;
}


template<typename T>
void Tensor<T>::zero() {

    std::fill(data.begin(), data.end(), static_cast<T>(0));
}

template<typename T>
T Tensor<T>::get(int i) const
{
    assert(num_dims == 1);
    return data[i];
}

template<typename T>
T Tensor<T>::get(int i, int j) const {

    assert(num_dims == 2);
    return data[j + dims[1] * i];
}

template<typename T>
T Tensor<T>::get(int i, int j, int k) const {

    assert(num_dims == 3);
    return data[k + dims[2] * (j + dims[1] * i)];
}

template<typename T>
T Tensor<T>::get(int i, int j, int k, int l) const {

    assert(num_dims == 4);
    return data[l + dims[3] * (k + dims[2] * (j + dims[1] * i))];
}

template<typename T>
void Tensor<T>::set(int i, T value) {

    assert(num_dims == 1);
    data[i] = value;
}

template<typename T>
void Tensor<T>::set(int i, int j, T value) {
    assert(num_dims == 2);
    data[j + dims[1] * i] = value;
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, T value) {
    assert(num_dims == 3);
    data[k + dims[2] * (j + dims[1] * i)] = value;
}

template<typename T>
void Tensor<T>::set(int i, int j, int k, int l, T value) {
    assert(num_dims == 4);
    data[l + dims[3] * (k + dims[2] * (j + dims[1] * i))] = value;
}

template<typename T>
void Tensor<T>::add(int i, T value) {
   assert(num_dims == 1);
    data[i] += value;
}

template<typename T>
void Tensor<T>::add(int i, int j, int k, int l, T value) {
    assert(num_dims == 4);
    data[l + dims[3] * (k + dims[2] * (j + dims[1] * i))] += value;
}

template <typename T>
void Tensor<T>::view(int new_num_dims, int* new_dims) {

    assert(new_num_dims > 0 && new_num_dims <= 4);
    this->num_dims = new_num_dims;
    std::copy(new_dims, new_dims+4, this->dims);
}

template <typename T>
Tensor<T> Tensor<T>::matmul(Tensor<T>& other) {
    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);

    int new_dims[] = {dims[0], other.dims[1]};
    Tensor<T> product(2, new_dims);
    for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<other.dims[1]; j++) {
            T sum = 0;
            for (int k=0; k<dims[1]; k++) {
                sum += get(i, k) * other.get(k, j);
            }
            product.set(i, j, sum);
        }
    }
    return product;
}

template <typename T>
Tensor<T> Tensor<T>::transpose() {
    assert(num_dims == 2);
    int new_dims[] = {dims[1], dims[0]};
    Tensor<T> result(2, new_dims);
    for (int i=0; i<dims[0]; i++) {
        for (int j=0; j<dims[1]; j++) {
            result.set(j, i, get(i, j));
        }
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::relu() {
    Tensor<T> result(num_dims, dims);
    for (int i=0; i<data.size(); i++) {
        result.data[i] = std::max(static_cast<T>(0), data[i]);
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() {
    Tensor<T> result(num_dims, dims);
    for (int i=0; i<data.size(); i++) {
        result.data[i] = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-data[i]));
    }
    return result;
}


