#include <cassert>
#include "host_matrix.h"

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


