// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Utility functions for optimizing multi-dimensional nonlinear functions.

#ifndef LIB_JXL_OPTIMIZE_H_
#define LIB_JXL_OPTIMIZE_H_

#include <cmath>
#include <cstdio>

#include "lib/jxl/base/status.h"

namespace jxl {
namespace optimize {

// An array type of numeric values that supports math operations with operator-,
// operator+, etc.
template <typename T, size_t N>
class Array {
 public:
  Array() = default;
  explicit Array(T v) {
    for (size_t i = 0; i < N; i++) v_[i] = v;
  }

  size_t size() const { return N; }

  T& operator[](size_t index) {
    JXL_DASSERT(index < N);
    return v_[index];
  }
  T operator[](size_t index) const {
    JXL_DASSERT(index < N);
    return v_[index];
  }

 private:
  // The values used by this Array.
  T v_[N];
};

template <typename T, size_t N>
Array<T, N> operator+(const Array<T, N>& x, const Array<T, N>& y) {
  Array<T, N> z;
  for (size_t i = 0; i < N; ++i) {
    z[i] = x[i] + y[i];
  }
  return z;
}

template <typename T, size_t N>
Array<T, N> operator-(const Array<T, N>& x, const Array<T, N>& y) {
  Array<T, N> z;
  for (size_t i = 0; i < N; ++i) {
    z[i] = x[i] - y[i];
  }
  return z;
}

template <typename T, size_t N>
Array<T, N> operator*(T v, const Array<T, N>& x) {
  Array<T, N> y;
  for (size_t i = 0; i < N; ++i) {
    y[i] = v * x[i];
  }
  return y;
}

template <typename T, size_t N>
T operator*(const Array<T, N>& x, const Array<T, N>& y) {
  T r = 0.0;
  for (size_t i = 0; i < N; ++i) {
    r += x[i] * y[i];
  }
  return r;
}

// Implementation of the Scaled Conjugate Gradient method described in the
// following paper:
//   Moller, M. "A Scaled Conjugate Gradient Algorithm for Fast Supervised
//   Learning", Neural Networks, Vol. 6. pp. 525-533, 1993
//   http://sci2s.ugr.es/keel/pdf/algorithm/articulo/moller1990.pdf
//
// The Function template parameter is a class that has the following method:
//
//   // Returns the value of the function at point w and sets *df to be the
//   // negative gradient vector of the function at point w.
//   double Compute(const optimize::Array<T, N>& w,
//                  optimize::Array<T, N>* df) const;
//
// Returns a vector w, such that |df(w)| < grad_norm_threshold.
template <typename T, size_t N, typename Function>
Array<T, N> OptimizeWithScaledConjugateGradientMethod(
    const Function& f, const Array<T, N>& w0, const T grad_norm_threshold,
    size_t max_iters) {
  const size_t n = w0.size();
  const T rsq_threshold = grad_norm_threshold * grad_norm_threshold;
  const T sigma0 = static_cast<T>(0.0001);
  const T l_min = static_cast<T>(1.0e-15);
  const T l_max = static_cast<T>(1.0e15);

  Array<T, N> w = w0;
  Array<T, N> wp;
  Array<T, N> r;
  Array<T, N> rt;
  Array<T, N> e;
  Array<T, N> p;
  T psq;
  T fp;
  T D;
  T d;
  T m;
  T a;
  T b;
  T s;
  T t;

  T fw = f.Compute(w, &r);
  T rsq = r * r;
  e = r;
  p = r;
  T l = static_cast<T>(1.0);
  bool success = true;
  size_t n_success = 0;
  size_t k = 0;

  while (k++ < max_iters) {
    if (success) {
      m = -(p * r);
      if (m >= 0) {
        p = r;
        m = -(p * r);
      }
      psq = p * p;
      s = sigma0 / std::sqrt(psq);
      f.Compute(w + (s * p), &rt);
      t = (p * (r - rt)) / s;
    }

    d = t + l * psq;
    if (d <= 0) {
      d = l * psq;
      l = l - t / psq;
    }

    a = -m / d;
    wp = w + a * p;
    fp = f.Compute(wp, &rt);

    D = 2.0 * (fp - fw) / (a * m);
    if (D >= 0.0) {
      success = true;
      n_success++;
      w = wp;
    } else {
      success = false;
    }

    if (success) {
      e = r;
      r = rt;
      rsq = r * r;
      fw = fp;
      if (rsq <= rsq_threshold) {
        break;
      }
    }

    if (D < 0.25) {
      l = std::min(4.0 * l, l_max);
    } else if (D > 0.75) {
      l = std::max(0.25 * l, l_min);
    }

    if ((n_success % n) == 0) {
      p = r;
      l = 1.0;
    } else if (success) {
      b = ((e - r) * r) / m;
      p = b * p + r;
    }
  }

  return w;
}

}  // namespace optimize
}  // namespace jxl

#endif  // LIB_JXL_OPTIMIZE_H_
