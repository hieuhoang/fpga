#pragma once

#include <iostream>
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define MAX_THREADS 512
#define MAX_BLOCKS 65535

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void HandleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

inline void HandleErrorCublas(cublasStatus_t err, const char *file, int line ) {
  if (err != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS ERROR: " << err << " in " << file << " at line " << line << std::endl;
    exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR_CUBLAS( err ) (HandleErrorCublas( err, __FILE__, __LINE__ ))

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define HANDLE_ERROR_CURAND(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }

inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
   if (stat != CURAND_STATUS_SUCCESS) {
      fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
   }
}
