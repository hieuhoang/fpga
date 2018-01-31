#pragma once
#include <cassert>
#include "types-fpga.h"
#include "host-matrix.h"

template<typename T>
class FPGAMatrix
{
public:
  FPGAMatrix(const OpenCLInfo &openCLInfo, MatrixIndexType indexType, unsigned a, unsigned b)
  :openCLInfo_(openCLInfo)
  ,indexType_(indexType)
  ,size_(a * b)
  {
    dim_[0] = a;
    dim_[1] = b;

    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_READ_WRITE,  sizeof(T) * size(), NULL, &err);
    CheckError(err);
  }

  FPGAMatrix(const OpenCLInfo &openCLInfo, MatrixIndexType indexType, const HostMatrix<T> &h_matrix)
  :openCLInfo_(openCLInfo)
  ,indexType_(indexType)
  ,size_(h_matrix.size())
  {
    dim_[0] = h_matrix.dim(0);
    dim_[1] = h_matrix.dim(1);

    std::vector<T> vec = h_matrix.Get(indexType_);

    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_COPY_HOST_PTR,  sizeof(T) * size(), (void*) vec.data(), &err);
    CheckError(err);

  }

  cl_mem &data()
  { return mem_; }

  const cl_mem &data() const
  { return mem_; }
 
  unsigned dim(unsigned i) const
  { return dim_[i]; }

  unsigned size() const
  { return size_; }


  void CopyTo(HostMatrix<T> &h_matrix) const
  {
    size_t bytes = size() * sizeof(T);

    std::vector<T> vec(size());

    CheckError( clEnqueueReadBuffer( openCLInfo_.commands, mem_, CL_TRUE, 0, sizeof(T) * size(), vec.data(), 0, NULL, NULL ) );
    CheckError( clFinish(openCLInfo_.commands) );

    h_matrix.CopyFrom(vec.data(), indexType_);

  }

protected:
  const OpenCLInfo &openCLInfo_;
  MatrixIndexType indexType_;
  unsigned dim_[2];
  unsigned size_;
  cl_mem mem_;
};

