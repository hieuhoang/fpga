#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <boost/timer/timer.hpp>
#include "host-matrix.h"
#include "const.h"

#ifndef NO_CL
#include "types-fpga.h"
#include "kernel.h"
#include "fpga-matrix.h"
#endif

#ifdef USE_CUDA
#include "cuda-code.h"
#endif

using namespace std;


int main()
{
  cerr << "Starting..." << endl;

  HostMatrix<float> h_W(VOCABSIZE, LAYER_DIM);
  HostMatrix<float> h_X(LAYER_DIM, MAXBATCH);
  HostMatrix<float> h_B(VOCABSIZE, 1);
  HostMatrix<MaxY> h_maxY(1, MAXBATCH);

  MaxY init;
  init.value = 3423;
  init.index = 9999;
  h_maxY.Set(init);

  ///*
  srand (time(NULL));
  Random(h_W);
  Random(h_X);
  Random(h_B);
  //*/
  /*
  h_W.Set(1);
  h_X.Set(1);
  h_B.Set(1);
  */

  boost::timer::cpu_timer timer;

#ifdef USE_CUDA
  cerr << "CUDA:" << endl;
  timer.start();
  RunCuda(h_maxY, h_W, h_X, h_B);
  cerr << "Operation took " << timer.format(2, "%w") << " sec" << endl;

  Debug(h_maxY);
#endif

#ifndef NO_CL
  cerr << "FPGA:" << endl;
  h_maxY.Set(init);


  OpenCLInfo openCLInfo;

  openCLInfo.context = CreateContext(100, openCLInfo.devices, openCLInfo.numDevices);
  cerr << "CreateContext done" << endl;

  openCLInfo.device = openCLInfo.devices[0];

  openCLInfo.commands = CreateCommandQueue(openCLInfo);
  cerr << "CreateCommandQueue done" << endl;

  CreateProgram(openCLInfo, "kernels/OutputLayer.aocx");
  cerr << "CreateProgram done" << endl;
  
  FPGAMatrix<MaxY> maxY(openCLInfo, rowMajor, 1, MAXBATCH);
  FPGAMatrix<float> W(openCLInfo, rowMajor, h_W);
  FPGAMatrix<float> X(openCLInfo, colMajor, h_X);
  FPGAMatrix<float> B(openCLInfo, rowMajor, h_B);
  
  cl_kernel kernel = CreateKernel("OutputLayer_float", openCLInfo);
  CallOpenCL(kernel, openCLInfo,
  			    W.data(), 
						X.data(), 
						B.data(), 
						maxY.data(),
						X.dim(1));
  CheckError( clFinish(openCLInfo.commands) );

  maxY.CopyTo(h_maxY);
  Debug(h_maxY);
#endif
  
  cerr << "HOST:" << endl;
  h_maxY.Set(init);
  HostMatrix<float> h_Y(VOCABSIZE, MAXBATCH);

  Affine(h_Y, h_W, h_X, h_B);
  Max(h_maxY, h_Y);
  Debug(h_maxY);

  cerr << "Finished" << endl;
}

