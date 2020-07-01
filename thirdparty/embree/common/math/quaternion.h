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

#include "vec3.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////
  // Quaternion Struct
  ////////////////////////////////////////////////////////////////

  template<typename T>
  struct QuaternionT
  {
    typedef Vec3<T> Vector;

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline QuaternionT           ()                     { }
    __forceinline QuaternionT           ( const QuaternionT& other ) { r = other.r; i = other.i; j = other.j; k = other.k; }
    __forceinline QuaternionT& operator=( const QuaternionT& other ) { r = other.r; i = other.i; j = other.j; k = other.k; return *this; }

    __forceinline          QuaternionT( const T& r       ) : r(r), i(zero), j(zero), k(zero) {}
    __forceinline explicit QuaternionT( const Vec3<T>& v ) : r(zero), i(v.x), j(v.y), k(v.z) {}
    __forceinline          QuaternionT( const T& r, const T& i, const T& j, const T& k ) : r(r), i(i), j(j), k(k) {}
    __forceinline          QuaternionT( const T& r, const Vec3<T>& v ) : r(r), i(v.x), j(v.y), k(v.z) {}

    __inline QuaternionT( const Vec3<T>& vx, const Vec3<T>& vy, const Vec3<T>& vz );
    __inline QuaternionT( const T& yaw, const T& pitch, const T& roll );

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline QuaternionT( ZeroTy ) : r(zero), i(zero), j(zero), k(zero) {}
    __forceinline QuaternionT( OneTy  ) : r( one), i(zero), j(zero), k(zero) {}

    /*! return quaternion for rotation around arbitrary axis */
    static __forceinline QuaternionT rotate(const Vec3<T>& u, const T& r) {
      return QuaternionT<T>(cos(T(0.5)*r),sin(T(0.5)*r)*normalize(u));
    }

    /*! returns the rotation axis of the quaternion as a vector */
    __forceinline Vec3<T> v( ) const { return Vec3<T>(i, j, k); }

  public:
    T r, i, j, k;
  };

  template<typename T> __forceinline QuaternionT<T> operator *( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a * b.r, a * b.i, a * b.j, a * b.k); }
  template<typename T> __forceinline QuaternionT<T> operator *( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r * b, a.i * b, a.j * b, a.k * b); }

  ////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////

  template<typename T> __forceinline QuaternionT<T> operator +( const QuaternionT<T>& a ) { return QuaternionT<T>(+a.r, +a.i, +a.j, +a.k); }
  template<typename T> __forceinline QuaternionT<T> operator -( const QuaternionT<T>& a ) { return QuaternionT<T>(-a.r, -a.i, -a.j, -a.k); }
  template<typename T> __forceinline QuaternionT<T> conj      ( const QuaternionT<T>& a ) { return QuaternionT<T>(a.r, -a.i, -a.j, -a.k); }
  template<typename T> __forceinline T              abs       ( const QuaternionT<T>& a ) { return sqrt(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }
  template<typename T> __forceinline QuaternionT<T> rcp       ( const QuaternionT<T>& a ) { return conj(a)*rcp(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }
  template<typename T> __forceinline QuaternionT<T> normalize ( const QuaternionT<T>& a ) { return a*rsqrt(a.r*a.r + a.i*a.i + a.j*a.j + a.k*a.k); }

  ////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////

  template<typename T> __forceinline QuaternionT<T> operator +( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a + b.r,  b.i,  b.j,  b.k); }
  template<typename T> __forceinline QuaternionT<T> operator +( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r + b, a.i, a.j, a.k); }
  template<typename T> __forceinline QuaternionT<T> operator +( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return QuaternionT<T>(a.r + b.r, a.i + b.i, a.j + b.j, a.k + b.k); }
  template<typename T> __forceinline QuaternionT<T> operator -( const T             & a, const QuaternionT<T>& b ) { return QuaternionT<T>(a - b.r, -b.i, -b.j, -b.k); }
  template<typename T> __forceinline QuaternionT<T> operator -( const QuaternionT<T>& a, const T             & b ) { return QuaternionT<T>(a.r - b, a.i, a.j, a.k); }
  template<typename T> __forceinline QuaternionT<T> operator -( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return QuaternionT<T>(a.r - b.r, a.i - b.i, a.j - b.j, a.k - b.k); }

  template<typename T> __forceinline Vec3<T>       operator *( const QuaternionT<T>& a, const Vec3<T>      & b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __forceinline QuaternionT<T> operator *( const QuaternionT<T>& a, const QuaternionT<T>& b ) {
    return QuaternionT<T>(a.r*b.r - a.i*b.i - a.j*b.j - a.k*b.k,
                          a.r*b.i + a.i*b.r + a.j*b.k - a.k*b.j,
                          a.r*b.j - a.i*b.k + a.j*b.r + a.k*b.i,
                          a.r*b.k + a.i*b.j - a.j*b.i + a.k*b.r);
  }
  template<typename T> __forceinline QuaternionT<T> operator /( const T             & a, const QuaternionT<T>& b ) { return a*rcp(b); }
  template<typename T> __forceinline QuaternionT<T> operator /( const QuaternionT<T>& a, const T             & b ) { return a*rcp(b); }
  template<typename T> __forceinline QuaternionT<T> operator /( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a*rcp(b); }

  template<typename T> __forceinline QuaternionT<T>& operator +=( QuaternionT<T>& a, const T             & b ) { return a = a+b; }
  template<typename T> __forceinline QuaternionT<T>& operator +=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a+b; }
  template<typename T> __forceinline QuaternionT<T>& operator -=( QuaternionT<T>& a, const T             & b ) { return a = a-b; }
  template<typename T> __forceinline QuaternionT<T>& operator -=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a-b; }
  template<typename T> __forceinline QuaternionT<T>& operator *=( QuaternionT<T>& a, const T             & b ) { return a = a*b; }
  template<typename T> __forceinline QuaternionT<T>& operator *=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a*b; }
  template<typename T> __forceinline QuaternionT<T>& operator /=( QuaternionT<T>& a, const T             & b ) { return a = a*rcp(b); }
  template<typename T> __forceinline QuaternionT<T>& operator /=( QuaternionT<T>& a, const QuaternionT<T>& b ) { return a = a*rcp(b); }

  template<typename T> __forceinline Vec3<T> xfmPoint ( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __forceinline Vec3<T> xfmVector( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __forceinline Vec3<T> xfmNormal( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a.r == b.r && a.i == b.i && a.j == b.j && a.k == b.k; }
  template<typename T> __forceinline bool operator !=( const QuaternionT<T>& a, const QuaternionT<T>& b ) { return a.r != b.r || a.i != b.i || a.j != b.j || a.k != b.k; }


  ////////////////////////////////////////////////////////////////////////////////
  /// Orientation Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> QuaternionT<T>::QuaternionT( const Vec3<T>& vx, const Vec3<T>& vy, const Vec3<T>& vz )
  {
    if ( vx.x + vy.y + vz.z >= T(zero) )
    {
      const T t = T(one) + (vx.x + vy.y + vz.z);
      const T s = rsqrt(t)*T(0.5f);
      r = t*s;
      i = (vy.z - vz.y)*s;
      j = (vz.x - vx.z)*s;
      k = (vx.y - vy.x)*s;
    }
    else if ( vx.x >= max(vy.y, vz.z) )
    {
      const T t = (T(one) + vx.x) - (vy.y + vz.z);
      const T s = rsqrt(t)*T(0.5f);
      r = (vy.z - vz.y)*s;
      i = t*s;
      j = (vx.y + vy.x)*s;
      k = (vz.x + vx.z)*s;
    }
    else if ( vy.y >= vz.z ) // if ( vy.y >= max(vz.z, vx.x) )
    {
      const T t = (T(one) + vy.y) - (vz.z + vx.x);
      const T s = rsqrt(t)*T(0.5f);
      r = (vz.x - vx.z)*s;
      i = (vx.y + vy.x)*s;
      j = t*s;
      k = (vy.z + vz.y)*s;
    }
    else //if ( vz.z >= max(vy.y, vx.x) )
    {
      const T t = (T(one) + vz.z) - (vx.x + vy.y);
      const T s = rsqrt(t)*T(0.5f);
      r = (vx.y - vy.x)*s;
      i = (vz.x + vx.z)*s;
      j = (vy.z + vz.y)*s;
      k = t*s;
    }
  }

  template<typename T> QuaternionT<T>::QuaternionT( const T& yaw, const T& pitch, const T& roll )
  {
    const T cya = cos(yaw  *T(0.5f));
    const T cpi = cos(pitch*T(0.5f));
    const T cro = cos(roll *T(0.5f));
    const T sya = sin(yaw  *T(0.5f));
    const T spi = sin(pitch*T(0.5f));
    const T sro = sin(roll *T(0.5f));
    r = cro*cya*cpi + sro*sya*spi;
    i = cro*cya*spi + sro*sya*cpi;
    j = cro*sya*cpi - sro*cya*spi;
    k = sro*cya*cpi - cro*sya*spi;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> static std::ostream& operator<<(std::ostream& cout, const QuaternionT<T>& q) {
    return cout << "{ r = " << q.r << ", i = " << q.i << ", j = " << q.j << ", k = " << q.k << " }";
  }

  /*! default template instantiations */
  typedef QuaternionT<float>  Quaternion3f;
  typedef QuaternionT<double> Quaternion3d;
}
