#pragma once

#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void HandleErrorCublas(cublasStatus_t err, const char *file, int line ) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS ERROR: " << err << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR_CUBLAS( err ) (HandleErrorCublas( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


