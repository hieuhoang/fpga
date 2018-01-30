#include <cstddef>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "cuda-code.h"

using namespace std;

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void HandleErrorCublas(cublasStatus_t err, const char *file, int line ) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS ERROR: " << err << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR_CUBLAS( err ) (HandleErrorCublas( err, __FILE__, __LINE__ ))

void runCuda()
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
  const float *A;
  const float *B;
  float *C;

  HANDLE_ERROR( cudaMalloc(&A, 85000 * 512 * sizeof(float)) );
  HANDLE_ERROR( cudaMalloc(&B, 512 * 640 * sizeof(float)) );
  HANDLE_ERROR( cudaMalloc(&C, 85000 * 640 * sizeof(float)) );

  HANDLE_ERROR_CUBLAS(cublasSgemm(handle, opA, opB,
                      m, n, k,
                      &alpha,
                      A, lda,
                      B, ldb,
                      &beta,
                      C, ldc));

}


