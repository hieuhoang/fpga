#pragma once
#include "host-matrix.h"

void RunCuda(HostMatrix<MaxY> &maxY, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B);

