#pragma once
#include <cassert>
#include <vector>
#include "const.h"

enum MatrixIndexType
{
   rowMajor, colMajor, fpgaSpecific
};

template<typename T>
class HostMatrix
{
public:
  HostMatrix(unsigned a, unsigned b)
  :size_(a * b)
  ,data_(a * b)
  {
    dim_[0] = a;
    dim_[1] = b;
  }

  const T *data() const
  { return data_.data(); }

  unsigned dim(unsigned i) const
  { return dim_[i]; }

  unsigned size() const
  { return size_; }

  const T &operator()(unsigned row, unsigned col) const
  {
    unsigned id = indices2Id(rowMajor, row, col);
    return data_[id];
  }

  T &operator()(unsigned row, unsigned col)
  {
    unsigned id = indices2Id(rowMajor, row, col);
    return data_[id];
  }

  std::vector<T> Get(MatrixIndexType indexType) const
  {
    std::vector<T> ret(size());
    assert(indexType != fpgaSpecific);

    for (unsigned row = 0; row < dim(0); ++row) {
      for (unsigned col = 0; col < dim(1); ++col) {
				unsigned id = indices2Id(indexType, row, col);
        ret[id] = (*this)(row, col);
      }
    }

    return ret;
  }

  void Set(const T &val)
  {
    for (unsigned i = 0; i < size(); ++i) {
      data_[i] = val;
    }
  }

  void CopyFrom(const T *arr, MatrixIndexType indexType)
  {
    for (unsigned row = 0; row < dim(0); ++row) {
      for (unsigned col = 0; col < dim(1); ++col) {
				unsigned id = indices2Id(indexType, row, col);
        const T &val = arr[id];
      
        (*this)(row, col) = val;
      }
    }
  }

  unsigned indices2Id(MatrixIndexType indexType, unsigned row, unsigned col) const
  {
    unsigned ind;

    assert(indexType != fpgaSpecific);
    assert(row < dim(0));
    assert(col < dim(1));

    if (indexType == colMajor) {
     ind = row + col * dim(0);
    }
    else {
     ind = row * dim(1) + col;
    }

    assert(ind < size());
    return ind;
  }

protected:
  unsigned dim_[2];
  unsigned size_;
	std::vector<T> data_;
};

/////////////////////////////////////////////////////

void Debug(HostMatrix<float> &matrix);
void Debug(HostMatrix<MaxY_type> &matrix);

void Affine(HostMatrix<float> &Y, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B);




