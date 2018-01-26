#pragma once
#include <cassert>
#include "types-fpga.h"
#include "host_matrix.h"

template<typename T>
class Matrix
{
public:
  Matrix(const OpenCLInfo &openCLInfo, bool rowMajor, unsigned a, unsigned b)
  :openCLInfo_(openCLInfo)
  ,rowMajor_(rowMajor)
  {
    dim_[0] = a;
    dim_[1] = b;
    size_ = a * b;

    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_READ_WRITE,  sizeof(T) * size(), NULL, &err);
    CheckError(err);
  }

  Matrix(const OpenCLInfo &openCLInfo, bool rowMajor, const HostMatrix<T> &h_matrix)
  :openCLInfo_(openCLInfo)
  ,rowMajor_(rowMajor)
  {
    dim_[0] = h_matrix.dim(0);
    dim_[1] = h_matrix.dim(1);
    size_ = h_matrix.size();

    cl_int err;
    mem_ = clCreateBuffer(openCLInfo.context,  CL_MEM_READ_WRITE,  sizeof(T) * size(), NULL, &err);
    CheckError(err);

  }

  cl_mem &data()
  { return mem_; }

  const cl_mem &data() const
  { return mem_; }

  bool isRowMajor() const
  { return rowMajor_; }
 
  unsigned dim(unsigned i) const
  { return dim_[i]; }

  unsigned size() const
  { return size_; }

  void CopyFrom(const T *arr, size_t count)
  {
    assert(count <= size_);
    size_t bytes = count * sizeof(T);
    CheckError( clEnqueueWriteBuffer(
                    openCLInfo_.commands,
                    mem_,
                    CL_TRUE,
                    0,
                    bytes,
                    arr,
                    0,
                    NULL,
                    NULL) );
    CheckError( clFinish(openCLInfo_.commands) );
  }

  void CopyTo(T *arr, size_t count) const
  {
    assert(count <= size_);
    size_t bytes = count * sizeof(T);

    CheckError( clEnqueueReadBuffer( openCLInfo_.commands, mem_, CL_TRUE, 0, sizeof(T) * size(), arr, 0, NULL, NULL ) );

    CheckError( clFinish(openCLInfo_.commands) );
  }

protected:
  const OpenCLInfo &openCLInfo_;
  bool rowMajor_;
  unsigned dim_[2];
  unsigned size_;
  cl_mem mem_;
};

