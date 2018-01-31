#pragma once
#include <cstddef>
#include "cuda-matrix.h"

// always column-major
template<typename T>
class CudaMatrixWrapper
{
public:
  CudaMatrixWrapper(const CudaMatrix<T> &other)
  :size_(other.size())
  ,data_(NULL)
  ,dataConst_(other.data())
  {
    dim_[0] = other.dim(0);
    dim_[1] = other.dim(1);
  }

  CudaMatrixWrapper(CudaMatrix<T> &other)
  :size_(other.size())
  ,data_(other.data())
  ,dataConst_(other.data())
  {
    dim_[0] = other.dim(0);
    dim_[1] = other.dim(1);
  }

  __device__ __host__
  unsigned dim(unsigned i) const
  { return dim_[i]; }

  __device__ __host__
  unsigned size() const
  { return size_; }

  __device__
  T* data()
  {
    assert(data_);
    return data_;
  }

  __device__
  const T* data() const
  {
    assert(dataConst_);
    return dataConst_;
  }

  __device__
  const T &operator[](unsigned i) const
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  T &operator[](unsigned i)
  {
    assert(i < size());
    return data()[i];
  }

  __device__
  inline const T &operator()(unsigned row, unsigned col) const
  {
    unsigned id = indices2Id(row, col);
    return data()[id];
  }

  __device__ __host__
  inline unsigned indices2Id(unsigned row, unsigned col) const
  {
    assert(row < dim(0));
    assert(col < dim(1));

    unsigned ind = row + col * dim(0);

    assert(ind < size());
    return ind;
  }

protected:
  unsigned dim_[2];
  unsigned size_;
  T *data_;
  const T *dataConst_;

};


