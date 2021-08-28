// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "math.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// RGB Color Class
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct Col3
  {
    T r, g, b;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Col3           ( )                   { }
    __forceinline Col3           ( const Col3& other ) { r = other.r; g = other.g; b = other.b; }
    __forceinline Col3& operator=( const Col3& other ) { r = other.r; g = other.g; b = other.b; return *this; }

    __forceinline explicit Col3 (const T& v)                         : r(v), g(v), b(v) {}
    __forceinline          Col3 (const T& r, const T& g, const T& b) : r(r), g(g), b(b) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Col3 (ZeroTy)   : r(zero)   , g(zero)   , b(zero)    {}
    __forceinline Col3 (OneTy)    : r(one)    , g(one)    , b(one)     {}
    __forceinline Col3 (PosInfTy) : r(pos_inf), g(pos_inf), b(pos_inf) {}
    __forceinline Col3 (NegInfTy) : r(neg_inf), g(neg_inf), b(neg_inf) {}
  };

  /*! output operator */
  template<typename T> __forceinline embree_ostream operator<<(embree_ostream cout, const Col3<T>& a) {
    return cout << "(" << a.r << ", " << a.g << ", " << a.b << ")";
  }

  /*! default template instantiations */
  typedef Col3<unsigned char> Col3uc;
  typedef Col3<float        > Col3f;
}
