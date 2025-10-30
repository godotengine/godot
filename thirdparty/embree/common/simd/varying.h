// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../sys/platform.h"

namespace embree
{
  /* Varying numeric types */
  template<int N>
  struct vfloat_impl
  {
    union { float f[N]; int i[N]; };
    __forceinline const float& operator [](size_t index) const { assert(index < N); return f[index]; }
    __forceinline       float& operator [](size_t index)       { assert(index < N); return f[index]; }
  };
  
  template<int N>
  struct vdouble_impl
  {
    union { double f[N]; long long i[N]; };
    __forceinline const double& operator [](size_t index) const { assert(index < N); return f[index]; }
    __forceinline       double& operator [](size_t index)       { assert(index < N); return f[index]; }
  };

  template<int N>
  struct vint_impl
  {
    int i[N];
    __forceinline const int& operator [](size_t index) const { assert(index < N); return i[index]; }
    __forceinline       int& operator [](size_t index)       { assert(index < N); return i[index]; }
  };
  
  template<int N>
  struct vuint_impl
  {
    unsigned int i[N];
    __forceinline const unsigned int& operator [](size_t index) const { assert(index < N); return i[index]; }
    __forceinline       unsigned int& operator [](size_t index)       { assert(index < N); return i[index]; }
  };

  template<int N>
  struct vllong_impl
  {
    long long i[N];
    __forceinline const long long& operator [](size_t index) const { assert(index < N); return i[index]; }
    __forceinline       long long& operator [](size_t index)       { assert(index < N); return i[index]; }
  };

  /* Varying bool types */
  template<int N> struct vboolf_impl { int       i[N]; }; // for float/int
  template<int N> struct vboold_impl { long long i[N]; }; // for double/long long
 
  /* Varying size constants */
#if defined(__AVX512VL__) // SKX
  const int VSIZEX = 8;  // default size
  const int VSIZEL = 16; // large size
#elif defined(__AVX__)
  const int VSIZEX = 8;
  const int VSIZEL = 8;
#else
  const int VSIZEX = 4;
  const int VSIZEL = 4;
#endif

  template<int N>
  struct vtypes {
    using vbool = vboolf_impl<N>;
    using vboolf = vboolf_impl<N>;
    using vboold = vboold_impl<N>;
    using vint = vint_impl<N>;
    using vuint = vuint_impl<N>;
    using vllong = vllong_impl<N>;
    using vfloat = vfloat_impl<N>;
    using vdouble = vdouble_impl<N>;
  };

  template<>
  struct vtypes<1> {
    using vbool = bool;
    using vboolf = bool;
    using vboold = bool;
    using vint = int;
    using vuint = unsigned int;
    using vllong = long long;
    using vfloat = float;
    using vdouble = double;
  };

  /* Aliases to default types */
  template<int N> using vbool = typename vtypes<N>::vbool;
  template<int N> using vboolf = typename vtypes<N>::vboolf;
  template<int N> using vboold = typename vtypes<N>::vboold;
  template<int N> using vint = typename vtypes<N>::vint;
  template<int N> using vuint = typename vtypes<N>::vuint;
  template<int N> using vllong = typename vtypes<N>::vllong;
  template<int N> using vreal = typename vtypes<N>::vfloat;
  template<int N> using vfloat = typename vtypes<N>::vfloat;
  template<int N> using vdouble = typename vtypes<N>::vdouble;

  /* 4-wide shortcuts */
  typedef vfloat<4>  vfloat4;
  typedef vdouble<4> vdouble4;
  typedef vreal<4>   vreal4;
  typedef vint<4>    vint4;
  typedef vuint<4>  vuint4;
  typedef vllong<4>  vllong4;
  typedef vbool<4>   vbool4;
  typedef vboolf<4>  vboolf4;
  typedef vboold<4>  vboold4;

  /* 8-wide shortcuts */
  typedef vfloat<8>  vfloat8;
  typedef vdouble<8> vdouble8;
  typedef vreal<8>   vreal8;
  typedef vint<8>    vint8;
  typedef vuint<8>    vuint8;
  typedef vllong<8>  vllong8;
  typedef vbool<8>   vbool8;
  typedef vboolf<8>  vboolf8;
  typedef vboold<8>  vboold8;

  /* 16-wide shortcuts */
  typedef vfloat<16>  vfloat16;
  typedef vdouble<16> vdouble16;
  typedef vreal<16>   vreal16;
  typedef vint<16>    vint16;
  typedef vuint<16>   vuint16;
  typedef vllong<16>  vllong16;
  typedef vbool<16>   vbool16;
  typedef vboolf<16>  vboolf16;
  typedef vboold<16>  vboold16;

  /* Default shortcuts */
  typedef vfloat<VSIZEX>  vfloatx;
  typedef vdouble<VSIZEX> vdoublex;
  typedef vreal<VSIZEX>   vrealx;
  typedef vint<VSIZEX>    vintx;
  typedef vuint<VSIZEX>   vuintx;
  typedef vllong<VSIZEX>  vllongx;
  typedef vbool<VSIZEX>   vboolx;
  typedef vboolf<VSIZEX>  vboolfx;
  typedef vboold<VSIZEX>  vbooldx;
}
