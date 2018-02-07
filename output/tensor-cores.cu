#include <curand.h>
#include <cublas_v2.h>
#include <mma.h>
#include "tensor-cores.h"
#include "types-cuda.h"

using namespace nvcuda;


// Must be multiples of 16 for wmma code to work
//#define MATRIX_M 16384
//#define MATRIX_N 16384
//#define MATRIX_K 16384
#define MATRIX_M 160
#define MATRIX_N 160
#define MATRIX_K 160


// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

////////////////////////////////////////////////////////////////////////////////////////////////


// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
   // Leading dimensions. Packed with no transpositions.
   int lda = M;
   int ldb = K;
   int ldc = M;

   // Tile using a 2D grid
   int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
   int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

   // Declare the fragments
   wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
   wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
   wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

   wmma::fill_fragment(acc_frag, 0.0f);

   // Loop over k
   for (int i = 0; i < K; i += WMMA_K) {
      int aRow = warpM * WMMA_M;
      int aCol = i;

      int bRow = i;
      int bCol = warpN * WMMA_N;

      // Bounds checking
      if (aRow < M && aCol < K && bRow < K && bCol < N) {
         // Load the inputs
         wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
         wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

         // Perform the matrix multiplication
         wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

      }
   }

   // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
   int cRow = warpM * WMMA_M;
   int cCol = warpN * WMMA_N;

   if (cRow < M && cCol < N) {
      wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


      for(int i=0; i < c_frag.num_elements; i++) {
         c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
      }

      // Store the output
      wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
   }
}


__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
   int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < n) {
      out[idx] = in[idx];
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////

void RunTensorCores()
{
  float *a_fp32;
   float *b_fp32;
   half *a_fp16;
   half *b_fp16;

   float *c;
   float *c_cublas;
   float *c_wmma;

   float *c_host_cublas;
   float *c_host_wmma;

   curandGenerator_t gen;
   cublasHandle_t cublasHandle;

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;

   cudaEvent_t startcublas;
   cudaEvent_t stopcublas;

   HANDLE_ERROR(cudaEventCreate(&startWMMA));
   HANDLE_ERROR(cudaEventCreate(&stopWMMA));

   HANDLE_ERROR(cudaEventCreate(&startcublas));
   HANDLE_ERROR(cudaEventCreate(&stopcublas));


   HANDLE_ERROR_CUBLAS(cublasCreate(&cublasHandle));

   // Use tensor cores
   HANDLE_ERROR_CUBLAS(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

   HANDLE_ERROR(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
   HANDLE_ERROR(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
   HANDLE_ERROR(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
   HANDLE_ERROR(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

   HANDLE_ERROR(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
   HANDLE_ERROR(cudaMalloc((void**)&c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));
   HANDLE_ERROR(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

   c_host_cublas = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));
   c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

   HANDLE_ERROR_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
   HANDLE_ERROR_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

   HANDLE_ERROR_CURAND(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
   HANDLE_ERROR_CURAND(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

   // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
   convertFp32ToFp16 <<< (MATRIX_M * MATRIX_K + 255) / 256, 256 >>> (a_fp16, a_fp32, MATRIX_M * MATRIX_K);
   convertFp32ToFp16 <<< (MATRIX_K * MATRIX_N + 255) / 256, 256 >>> (b_fp16, b_fp32, MATRIX_K * MATRIX_N);

   HANDLE_ERROR_CURAND(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

   HANDLE_ERROR_CURAND(curandDestroyGenerator(gen));

   HANDLE_ERROR(cudaMemcpy(c_cublas, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));
   HANDLE_ERROR(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

   float alpha = 2.0f;
   float beta = 2.0f;


   printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

   // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   printf("Running with wmma...\n");
   HANDLE_ERROR(cudaEventRecord(startWMMA));
   wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);
   HANDLE_ERROR(cudaEventRecord(stopWMMA));



   // Now using cuBLAS
   printf("Running with cuBLAS...\n");
   HANDLE_ERROR(cudaEventRecord(startcublas));
   HANDLE_ERROR_CUBLAS(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                MATRIX_M, MATRIX_N, MATRIX_K,
                &alpha,
                a_fp16, CUDA_R_16F, MATRIX_M,
                b_fp16, CUDA_R_16F, MATRIX_K,
                &beta,
                c_cublas, CUDA_R_32F, MATRIX_M,
                CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));
   HANDLE_ERROR(cudaEventRecord(stopcublas));

   // Error checking
   printf("\nChecking results...\n");
   HANDLE_ERROR(cudaMemcpy(c_host_wmma, c_wmma, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));
   HANDLE_ERROR(cudaMemcpy(c_host_cublas, c_cublas, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToHost));

   // 0.01% relative tolerance. 1e-5 absolute tolerance.
   int errors = 0;
   for (int i = 0; i < MATRIX_M * MATRIX_N; i++) {
      float v1 = c_host_wmma[i];
      float v2 = c_host_cublas[i];
      if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
         errors++;
         if (errors < 10) printf("%f %f\n", v1, v2);
      }
   }

   if (errors > 0) {
      printf("WMMA does not agree with cuBLAS! %d errors!\n", errors);
   }
   else {
      printf("Results verified: cublas and WMMA agree.\n\n");
      float wmmaTime;
      float cublasTime;
      HANDLE_ERROR(cudaEventSynchronize(stopWMMA));
      HANDLE_ERROR(cudaEventSynchronize(stopcublas));
      HANDLE_ERROR(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
      HANDLE_ERROR(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
      printf("wmma took %fms\n", wmmaTime);
      printf("cublas took %fms\n", cublasTime);

      printf("\nFor a faster code using wmma you should check out the cudaTensorCoreGemm sample in the CUDA Toolkit.\nThis code was written as a demo only!\n\n");
   }


   HANDLE_ERROR(cudaEventDestroy(startWMMA));
   HANDLE_ERROR(cudaEventDestroy(stopWMMA));

   HANDLE_ERROR(cudaEventDestroy(startcublas));
   HANDLE_ERROR(cudaEventDestroy(stopcublas));

   HANDLE_ERROR(cudaFree(a_fp32));
   HANDLE_ERROR(cudaFree(b_fp32));
   HANDLE_ERROR(cudaFree(a_fp16));
   HANDLE_ERROR(cudaFree(b_fp16));

   HANDLE_ERROR(cudaFree(c));
   HANDLE_ERROR(cudaFree(c_cublas));
   HANDLE_ERROR(cudaFree(c_wmma));

   free(c_host_cublas);
   free(c_host_wmma);

   HANDLE_ERROR(cudaDeviceReset());

}
