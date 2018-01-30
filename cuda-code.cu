#include <cstddef>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include "cuda-code.h"

using namespace std;

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

  int m = 7;
  int n = 7;
  int k = 7;

  int lda = 4;
  int ldb = 6;
  int ldc = 7;

  const float *alpha = NULL;
  const float *beta = NULL;
  const float *A = NULL;
  const float *B = NULL;
  float *C = NULL;

  HANDLE_ERROR_CUBLAS(cublasSgemm(handle, opB, opA,
                      n, m, k,
                      alpha,
                      B, ldb,
                      A, lda,
                      beta,
                      C, ldc));

}


