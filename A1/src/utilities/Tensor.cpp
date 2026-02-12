#include "Tensor.hpp"

template
class Tensor<int>;

template
class Tensor<float>;

template
class Tensor<double>;


template<typename T>
Tensor<T>::Tensor(int num_dims, std::vector<int> dims) {

    assert(num_dims > 0 && num_dims <= 4);
    int size = 1;
    for (int i=0; i<num_dims; i++){
        size *= dims[i];
        this->dims[i] = dims[i];
    }
    data.resize(size);
    this->num_dims = num_dims;
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T> &other) : data(other.data), num_dims(other.num_dims), dims(other.dims){}

template<typename T>
void Tensor<T>::zero() {
    std::fill(data.begin(), data.end(), T(0));
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
void Tensor<T>::view(int new_num_dims, std::vector<int> new_dims) {

    assert(new_num_dims > 0 && new_num_dims <= 4);
    this->num_dims = new_num_dims;
    this->dims = std::move(new_dims);
}

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) {
    assert(num_dims == 2 && other.num_dims == 2);
    assert(dims[1] == other.dims[0]);

    std::vector<int> new_dims = {dims[0], other.dims[1]};
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
    std::vector<int> new_dims = {dims[1], dims[0]};
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
        result.data[i] = std::max(T(0), data[i]);
    }
    return result;
}

template<typename T>
T sigmoid(T x) {
    return T(1) / (T(1) + std::exp(-x));
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid() {
    Tensor<T> result(num_dims, dims);
    for (int i=0; i<data.size(); i++) {
        result.data[i] = ::sigmoid(data[i]);
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::softmax() {
    assert(num_dims == 2);
    const int rows = dims[0], cols = dims[1];
    Tensor<T> probabilities(2,dims);
    for (int i=0; i<rows; i++) {
        T row_max = get(i,0);
        for (int j=1; j<cols; j++) {
            if (get(i,j) > row_max) {
                row_max = get(i,j);
            }
        }
        T denominator = T(0);
        for (int j=0; j<cols; j++) {
            T val = get(i,j);
            denominator += std::exp(val-row_max);
        }
        for (int j=0; j<cols; j++) {
            probabilities.set(i,j,std::exp(get(i,j)-row_max)/denominator);
        }
    }
    return probabilities;
}



template <typename T>
Tensor<T> Tensor<T>::sigmoid_derivative() {
    Tensor<T> result(num_dims, dims);
    for (int i=0; i<data.size(); i++) {
        result.data[i] = ::sigmoid(data[i]) * (T(1) - ::sigmoid(data[i]));
    }
    return result;
}

template <typename T>
Tensor<T> Tensor<T>::relu_derivative() {
    Tensor<T> result(num_dims, dims);
    for (int i=0; i<data.size(); i++) {
        result.data[i] = data[i] > T(0) ? T(1) : T(0);
    }
    return result;
}

template <typename T>
T Tensor<T>::sum() const {
    return std::accumulate(data.begin(), data.end(), T(0));
}

template <typename T>
std::string print_vector(std::vector<T> v) {
    std::string ans = "[";
    for (auto elem: v) {
        ans += std::to_string(elem) + " ";
    }
    ans += "]";
    return ans;
}

template <typename T>
Tensor<T> Tensor<T>::operator+(Tensor<T>& other) {
    if (other.num_dims == 1 && this->num_dims == 2 && other.data.size() == this->dims[1]) {
        Tensor<T> sum(num_dims,dims);
        for (int i=0; i < this->dims[0]; i++) {
            for (int j=0; j< this->dims[1]; j++) {
                sum.set(i,j,get(i,j)+other.get(j));
            }
        }
        return sum;
    }
    else if (other.dims == this->dims) {
        Tensor<T> sum(num_dims,other.dims);
        for (int i=0; i <data.size(); i++) {
            sum.data[i] = data[i] + other.data[i];
        }
        return sum;
    }
    else {
        throw std::logic_error("Adding Tensor of dim" + print_vector(other.dims) + "to Tensor of dim" + print_vector(this->dims)+ "is not allowed");
    }
}

template<typename T>
Tensor<T> Tensor<T>::operator*(Tensor<T> &other) {
    assert(dims == other.dims);
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < data.size(); ++i) {
        product.data[i] = data[i] * other.data[i];
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator*(T scalar) {
    Tensor<T> product(num_dims, dims);
    for (int i = 0; i < data.size(); i++) {
        product.data[i] = data[i] * scalar;
    }
    return product;
}

template<typename T>
Tensor<T> Tensor<T>::operator/(T scalar) {
    if (scalar == T(0)) {
        throw std::logic_error("Division by zero");
    }
    Tensor<T> result(num_dims, dims);
    for (int i = 0; i < data.size(); i++) {
        result.data[i] = data[i] / scalar;
    }
    return result;
}

template<typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T> &other) {
    assert(this->dims == other.dims);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = data[i] - other.data[i];
    }
    return *this;
}

template<typename T>
Tensor<T> Tensor<T>::col_sum() const{
    assert(num_dims == 2);
    int rows = dims[0], cols = dims[1];
    std::vector<int> sum_dims = {cols};
    Tensor<T> sum(1, sum_dims);
    for (int i = 0; i < cols; ++i) {
        T total = 0;
        for (int j = 0; j < rows; ++j) {
            total += get(j, i);
        }
        sum.set(i, total);
    }
    return sum;
}

template<>
void Tensor<double>::randn(std::default_random_engine generator, std::normal_distribution<double> distribution, double multiplier) {
    for (double & i : data) {
        i = distribution(generator) * multiplier;
    }
}

template<>
void Tensor<double>::print() const{
    if (num_dims == 2) {
        int rows = dims[0], cols = dims[1];
        std::cout << "Tensor2D (" << rows << ", " << cols << ")\n[";
        for (int i = 0; i < rows; ++i) {
            if (i != 0) std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < cols; ++j) {
                if (j == (cols - 1)) {
                    printf("%.18lf", get(i, j));
                } else {
                    printf("%.18lf ", get(i, j));
                }

            }
            if (i == (rows - 1)) {
                std::cout << "]]\n";
            } else {
                std::cout << "]\n";
            }
        }
    } else {
        printf("Tensor%dd (", num_dims);
        for (int i = 0; i < num_dims; ++i) {
            printf("%d", dims[i]);
            if (i != (num_dims - 1)) {
                printf(",");
            }
        }
        printf(")\n[");
        for (double j : data) {
            printf("%lf ", j);
        }
        printf("]\n");
    }
}

template<typename T>
void Tensor<T>::dropout(std::default_random_engine generator, std::uniform_real_distribution<> distribution, double p) {
    for (T &i : data) {
        i = (distribution(generator) < p) / p;
    }
}

template<typename T>
Tensor<T> Tensor<T>::convolve2D(Tensor<T> &kernels, int stride, int padding, Tensor<T> bias) {
    assert(kernels.dims[1] == dims[1]); // matching number of channels in kernels and input tensor
    int new_h = ((dims[2] + 2*padding - kernels.dims[2])/stride ) + 1;
    int new_w = ((dims[3] + 2*padding - kernels.dims[3])/stride ) + 1;

    std::vector<int> result_dims = {dims[0], kernels.dims[0], new_h, new_w};
    Tensor<T> result(4,result_dims);
    for (int i=0; i<dims[0]; i++) {
        for (int j=0; j < kernels.dims[0]; j++) {
            for (int k = 0; k < new_h; k++) {
                for (int l=0; l < new_w; l++) {
                    int im_si = stride * k - padding;
                    int im_sj = stride * l - padding;
                    T total = 0;
                    for (int m=0; m < kernels.dims[1]; m++) {
                        for (int n=0; n < kernels.dims[2]; n++) {
                            for (int o=0; o < kernels.dims[3]; o++) {
                                int x = im_si + n, y = im_sj + o;
                                if (x < 0 || x >= dims[2] || y < 0 || y >= dims[3]) {
                                    continue;
                                }
                                T a = get(i,m,x,y);
                                T b = kernels.get(j,m, n, o);
                                total += a * b;
                            }
                        }
                    }
                    result.set(i,j,k,l,total+bias.get(j));
                }
            }
        }
    }
    return result;
}



