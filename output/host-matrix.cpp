#include <iostream>
#include <cassert>
#include "host-matrix.h"

using namespace std;

void Debug(HostMatrix<float> &matrix)
{
  for (unsigned row = 0; row < matrix.dim(0); ++row) {
    for (unsigned col = 0; col < matrix.dim(1); ++col) {
      cerr << matrix(row, col) << " ";
    }
    cerr << endl;
  }  
}

void Debug(HostMatrix<MaxY> &matrix)
{
  for (unsigned row = 0; row < matrix.dim(0); ++row) {
    for (unsigned col = 0; col < matrix.dim(1); ++col) {
      const MaxY &val = matrix(row, col);
      cerr << "(" << val.value << "," << val.index << ") ";
    }
    cerr << endl;
  }  
}

void Random(HostMatrix<float> &matrix)
{
  for (unsigned i = 0; i < matrix.size(); ++i) {
    matrix[i] = ((double) rand() / (RAND_MAX));
  }
}

void Affine(HostMatrix<float> &Y, const HostMatrix<float> &W, const HostMatrix<float> &X, const HostMatrix<float> &B)
{
  assert(Y.dim(0) == W.dim(0));
  assert(Y.dim(1) == X.dim(1));

  assert(W.dim(1) == X.dim(0));
  assert(W.dim(0) == B.dim(0));
  assert(B.dim(1) == 1);

  for (unsigned rowW = 0; rowW < W.dim(0); ++rowW) {
    for (unsigned colX = 0; colX < X.dim(1); ++colX) {
      float sum = 0;
      for (unsigned colW = 0; colW < W.dim(1); ++colW) {
        sum += W(rowW, colW) * X(colW, colX);
      }
      Y(rowW, colX) = sum + B(rowW, 0);
    }
  }
}

void Max(HostMatrix<MaxY> &maxY, const HostMatrix<float> &Y)
{
  assert(maxY.dim(0) == 1);
  assert(maxY.dim(1) == Y.dim(1));

  for (unsigned col = 0; col < Y.dim(1); ++col) {
    float max = Y(0, col);
    unsigned argmax = 0;
    for (unsigned row = 1; row < Y.dim(0); ++row) {
      if (max < Y(row, col)) {
        max = Y(row, col);
        argmax = row;
      }
    }

    MaxY &ele = maxY[col];
    ele.value = max;
    ele.index = argmax;
  }
}


