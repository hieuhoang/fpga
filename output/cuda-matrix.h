#pragma once
#include <cassert>
#include "host-matrix.h"
#include "types-cuda.h"

// always column-major
template<typename T>
class CudaMatrix
{
public:
  CudaMatrix() = delete;
  CudaMatrix(const CudaMatrix&) = delete;

  CudaMatrix(unsigned a, unsigned b)
  :size_(a * b)
  ,data_(NULL)
  {
    dim_[0] = a;
    dim_[1] = b;

    HANDLE_ERROR(cudaMalloc(&data_, size_ * sizeof(T)));
  }

  CudaMatrix(const HostMatrix<T> &h_matrix)
  :size_(h_matrix.size())
  ,data_(NULL)
  {
    dim_[0] = h_matrix.dim(0);
    dim_[1] = h_matrix.dim(1);

    std::vector<T> vec = h_matrix.Get(colMajor);

    size_t bytes = size_ * sizeof(T);
    HANDLE_ERROR(cudaMalloc(&data_, bytes));
    HANDLE_ERROR(cudaMemcpy(data_, vec.data(), bytes, cudaMemcpyHostToDevice));

  }

  T *data()
  { return data_; }

  const T *data() const
  { return data_; }

  unsigned dim(unsigned i) const
  { return dim_[i]; }

  unsigned size() const
  { return size_; }

  void CopyTo(HostMatrix<T> &h_matrix) const
  {
    size_t bytes = size() * sizeof(T);

    std::vector<T> vec(size());
    std::cerr << "vec.data()="
              << vec.data() << " "
              << data_ << " "
              << bytes << " "
              <<  std::endl;

    HANDLE_ERROR(cudaMemcpy(vec.data(), data_, bytes, cudaMemcpyDeviceToHost));

    h_matrix.CopyFrom(vec.data(), colMajor);

  }

protected:
  unsigned dim_[2];
  unsigned size_;
  T *data_;
};

