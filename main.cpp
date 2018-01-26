#include <iostream>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "types-fpga.h"
#include "kernel.h"
#include "matrix.h"

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

  Matrix<float> W(openCLInfo, true, 85000, 512);
  Matrix<float> X(openCLInfo, true, 512, 640);
  Matrix<float> B(openCLInfo, true, 1, 85000);
  Matrix<float> Y(openCLInfo, true, 85000, 640);

  vector<float> vec;
  
  cerr << "main1" << endl;
  vec.resize(W.size(), 3.3);
  W.CopyFrom(vec.data(), vec.size());

  vec.resize(X.size(), 21.2);
  X.CopyFrom(vec.data(), vec.size());

  vec.resize(B.size(), 9.3443);
  B.CopyFrom(vec.data(), vec.size());

  cerr << "main2" << endl;

  cl_kernel kernel = CreateKernel("OutputLayer_float", openCLInfo);

  for (size_t i = 0; i < 1; ++i) {
    //CallOpenCL("OutputLayer_float", openCLInfo,
    //    W.data(), X.data(), B.data(), Y.data(), X.dim(1));

    CallOpenCL(kernel, openCLInfo,
        W.data(), X.data(), B.data(), Y.data(), X.dim(1));
    CheckError( clFinish(openCLInfo.commands) );

  }

  vec.resize(Y.size());
  Y.CopyTo(vec.data(), vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    cerr << vec[i] << " ";
  }  
  cerr << endl;

  cerr << "Finished" << endl;
}

