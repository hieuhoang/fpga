#include <cstddef>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "cuda-code.h"
#include "cuda-matrix.h"
#include "types-cuda.h"

using namespace std;

void runCuda(HostMatrix<MaxY_type> &maxY, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B)
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

  int m = 85000;
  int n = 640;
  int k = 512;

  int lda = 85000;
  int ldb = 512;
  int ldc = 85000;

  const float alpha = 1;
  const float beta = 0;

  CudaMatrix<float> cudaW(W);
  CudaMatrix<float> cudaX(X);
  CudaMatrix<float> cudaB(B);
  CudaMatrix<float> cudaY(85000, 640);

  HANDLE_ERROR_CUBLAS(cublasSgemm(handle, opA, opB,
                      m, n, k,
                      &alpha,
                      cudaW.data(), lda,
                      cudaX.data(), ldb,
                      &beta,
                      cudaY.data(), ldc));

}


