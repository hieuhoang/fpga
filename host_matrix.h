#pragma once
#include <cassert>
#include <vector>

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

  std::vector<T> Get(MatrixIndexType indexType)
  {
    std::vector<T> ret(size());
    assert(indexType != fpgaSpecific);


    return ret;
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

  void id2Indices(MatrixIndexType indexType, unsigned id, unsigned *out) const
  {
    assert(id < size());

    if (indexType == colMajor) {
      out[0] = id / dim(0);
      out[1] = id % dim(0);
    }
    else if (indexType == rowMajor) {
      out[1] = id / dim(1);
      out[0] = id % dim(1);
    }
    else {
      assert(false);
    }

    assert(out[0] < dim(0));
    assert(out[1] < dim(1));
  }

protected:
  unsigned dim_[2];
  unsigned size_;
	std::vector<float> data_;
};


