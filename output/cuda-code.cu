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
void gCalcMax(CudaMatrixWrapper<MaxY> out, const CudaMatrixWrapper<float> in)
{
  extern __shared__ MaxY tmp[];
  CudaMatrixWrapper<MaxY> maxes(tmp, blockDim.x, 1);

  assert(out.dim(1) == in.dim(1));
  unsigned rows = in.dim(0);

  unsigned col = blockIdx.x; // hypoInd
  assert(col < in.dim(1));

  unsigned row = threadIdx.x; // vocabInd
  assert(row < rows);

  MaxY &ele = maxes[threadIdx.x];
  ele.value = in(row, col);
  ele.index = row;

  row += blockDim.x;
  while (row < rows) {
    float val = in(row, col);
    if (val > ele.value) {
      ele.value = val;
      ele.index = row;
    }

    row += blockDim.x;
  }

  if (threadIdx.x == 0) {
    for (unsigned i = 1; i < blockDim.x; ++i) {
      if (maxes[0].value < maxes[i].value) {
        maxes[0].value = maxes[i].value;
        maxes[0].index = maxes[i].index;
      }
    }

    out[col].value = maxes[0].value;
    out[col].index = maxes[0].index;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

void RunCuda(HostMatrix<MaxY> &maxY, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B)
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

  CudaMatrix<MaxY> cudaMaxY(1, MAXBATCH);

  unsigned blocks = std::min((unsigned) MAX_BLOCKS, cudaY.dim(1));
  unsigned threads = 1; // std::min((unsigned)MAX_THREADS, cudaY.dim(1));
  unsigned shared = sizeof(MaxY) * threads;

  cerr << "blocks=" << blocks << " threads=" << threads << endl;

  gCalcMax<<<blocks, threads, shared>>>(cudaMaxY, cudaY);

  cudaDeviceSynchronize();

  cudaMaxY.CopyTo(maxY);
}


