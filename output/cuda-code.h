#pragma once
#include "host-matrix.h"

void runCuda(HostMatrix<MaxY_type> &maxY, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B);
