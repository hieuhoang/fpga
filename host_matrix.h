#pragma once
#include <cassert>
#include <vector>

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

  const T &operator()(unsigned a, unsigned b) const
  {
    unsigned id = a + b * dim_[0];
    return data_[id];
  }

  T &operator()(unsigned a, unsigned b)
  {
    unsigned id = a + b * dim_[0];
    return data_[id];
  }

protected:
  unsigned dim_[2];
  unsigned size_;
	std::vector<float> data_;
};


