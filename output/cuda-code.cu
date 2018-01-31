#include <cstddef>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "cuda-code.h"
#include "cuda-matrix.h"
#include "cuda-matrix-wrapper.h"
#include "types-cuda.h"
#include "const.h"

using namespace std;

__global__
void gCalcMax(CudaMatrixWrapper<MaxY_type> out, const CudaMatrixWrapper<float> in)
{
  assert(out.dim(1) == in.dim(1));
  for (unsigned col = 0; col < in.dim(1); ++col) {
    unsigned maxIndex = 0;
    float value = in(0, col);

    for (unsigned row = 1; row < in.dim(0); ++row) {
      float val = in(row, col);
      if (val > value) {
        value = val;
        maxIndex = row;
      }
    }

    MaxY_type &ele = out[col];
    ele.value = value;
    ele.index = maxIndex;
  }
}

void RunCuda(HostMatrix<MaxY_type> &maxY, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B)
{ 
  cublasHandle_t handle;
  cublasStatus_t stat;
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublasCreate initialization failed" << std::endl;
    abort();
  }

  cublasOperation_t opA = CUBLAS_OP_N;
  cublasOperation_t opB = CUBLAS_OP_N;

  int m = VOCABSIZE;
  int n = MAXBATCH;
  int k = LAYER_DIM;

  int lda = VOCABSIZE;
  int ldb = LAYER_DIM;
  int ldc = VOCABSIZE;

  const float alpha = 1;
  const float beta = 0;

  CudaMatrix<float> cudaW(W);
  CudaMatrix<float> cudaX(X);
  CudaMatrix<float> cudaB(B);
  CudaMatrix<float> cudaY(VOCABSIZE, MAXBATCH);

  HANDLE_ERROR_CUBLAS(cublasSgemm(handle, opA, opB,
                      m, n, k,
                      &alpha,
                      cudaW.data(), lda,
                      cudaX.data(), ldb,
                      &beta,
                      cudaY.data(), ldc));

  CudaMatrix<MaxY_type> cudaMaxY(1, MAXBATCH);
  gCalcMax<<<1,1>>>(cudaMaxY, cudaY);

  cudaDeviceSynchronize();

  cudaMaxY.CopyTo(maxY);
}


