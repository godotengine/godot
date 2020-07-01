// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
  template<typename T> inline std::ostream& operator<<(std::ostream& cout, const Col3<T>& a) {
    return cout << "(" << a.r << ", " << a.g << ", " << a.b << ")";
  }

  /*! default template instantiations */
  typedef Col3<unsigned char> Col3uc;
  typedef Col3<float        > Col3f;
}
