#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "types-fpga.h"
#include "kernel.h"
#include "matrix.h"
#include "host_matrix.h"
#include "const.h"


using namespace std;


int main()
{
  cerr << "Starting..." << endl;

  OpenCLInfo openCLInfo;

  openCLInfo.context = CreateContext(100, openCLInfo.devices, openCLInfo.numDevices);
  cerr << "CreateContext done" << endl;

  openCLInfo.device = openCLInfo.devices[0];

  openCLInfo.commands = CreateCommandQueue(openCLInfo);
  cerr << "CreateCommandQueue done" << endl;

  CreateProgram(openCLInfo, "kernels/fpga.aocx");
  cerr << "CreateProgram done" << endl;

  HostMatrix<float> h_W(VOCABSIZE, LAYER_DIM);
  HostMatrix<float> h_X(LAYER_DIM, 640);
  HostMatrix<float> h_B(VOCABSIZE, 1);
  HostMatrix<float> h_Y(VOCABSIZE, 640);
  HostMatrix<MaxY_type> h_maxY(1, 640);

  h_W.Set(43.232);
  h_X.Set(67.2);
  h_B.Set(125.87);
  h_Y.Set(8.55);

  Matrix<float> W(openCLInfo, rowMajor, h_W);
  Matrix<float> X(openCLInfo, colMajor, h_X);
  Matrix<float> B(openCLInfo, rowMajor, h_B);
  Matrix<MaxY_type> maxY(openCLInfo, rowMajor, 1, 640);

  cerr << "FPGA:" << endl;
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

  cerr << "HOST:" << endl;
  Affine(h_Y, h_W, h_X, h_B);

  Debug(h_Y);

  cerr << "Finished" << endl;
}

