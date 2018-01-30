#pragma once
#include <cassert>
#include "host-matrix.h"

// always column-major
template<typename T>
class CudaMatrix
{
public:
  CudaMatrix(unsigned a, unsigned b)
  :size_(a * b)
  {
    dim_[0] = a;
    dim_[1] = b;
    cudaMalloc(&data_, size_ * sizeof(T));
  }

  CudaMatrix(MatrixIndexType indexType, const HostMatrix<T> &h_matrix)
  :size_(h_matrix.size())
  {
    dim_[0] = h_matrix.dim(0);
    dim_[1] = h_matrix.dim(1);

    std::vector<T> vec = h_matrix.Get(colMajor);

  }

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

    h_matrix.CopyFrom(vec.data(), colMajor);

  }

protected:
  unsigned dim_[2];
  unsigned size_;
  T *data_;
};

