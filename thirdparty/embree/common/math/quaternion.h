// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec3.h"
#include "vec4.h"

#include "transcendental.h"

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
    __forceinline explicit QuaternionT( const Vec4<T>& v ) : r(v.x), i(v.y), j(v.z), k(v.w) {}
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

  // evaluates a*q-r
  template<typename T> __forceinline QuaternionT<T>
  msub(const T& a, const QuaternionT<T>& q, const QuaternionT<T>& p)
  {
    return QuaternionT<T>(msub(a, q.r, p.r),
                          msub(a, q.i, p.i),
                          msub(a, q.j, p.j),
                          msub(a, q.k, p.k));
  }
  // evaluates a*q-r
  template<typename T> __forceinline QuaternionT<T>
  madd (const T& a, const QuaternionT<T>& q, const QuaternionT<T>& p)
  {
    return QuaternionT<T>(madd(a, q.r, p.r),
                          madd(a, q.i, p.i),
                          madd(a, q.j, p.j),
                          madd(a, q.k, p.k));
  }

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

  template<typename T, typename M> __forceinline QuaternionT<T>
  select(const M& m, const QuaternionT<T>& q, const QuaternionT<T>& p)
  {
    return QuaternionT<T>(select(m, q.r, p.r),
                          select(m, q.i, p.i),
                          select(m, q.j, p.j),
                          select(m, q.k, p.k));
  }


  template<typename T> __forceinline Vec3<T> xfmPoint ( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __forceinline Vec3<T> xfmVector( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }
  template<typename T> __forceinline Vec3<T> xfmNormal( const QuaternionT<T>& a, const Vec3<T>&       b ) { return (a*QuaternionT<T>(b)*conj(a)).v(); }

  template<typename T> __forceinline T dot(const QuaternionT<T>& a, const QuaternionT<T>& b) { return a.r*b.r + a.i*b.i + a.j*b.j + a.k*b.k; }

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

  //////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  //////////////////////////////////////////////////////////////////////////////

  template<typename T> static embree_ostream operator<<(embree_ostream cout, const QuaternionT<T>& q) {
    return cout << "{ r = " << q.r << ", i = " << q.i << ", j = " << q.j << ", k = " << q.k << " }";
  }

  /*! default template instantiations */
  typedef QuaternionT<float>  Quaternion3f;
  typedef QuaternionT<double> Quaternion3d;

  template<int N> using Quaternion3vf = QuaternionT<vfloat<N>>;
  typedef QuaternionT<vfloat<4>>  Quaternion3vf4;
  typedef QuaternionT<vfloat<8>>  Quaternion3vf8;
  typedef QuaternionT<vfloat<16>> Quaternion3vf16;

  //////////////////////////////////////////////////////////////////////////////
  /// Interpolation
  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  __forceinline QuaternionT<T>lerp(const QuaternionT<T>& q0,
                                   const QuaternionT<T>& q1,
                                   const T& factor)
  {
    QuaternionT<T> q;
    q.r = lerp(q0.r, q1.r, factor);
    q.i = lerp(q0.i, q1.i, factor);
    q.j = lerp(q0.j, q1.j, factor);
    q.k = lerp(q0.k, q1.k, factor);
    return q;
  }

  template<typename T>
  __forceinline QuaternionT<T> slerp(const QuaternionT<T>& q0,
                                     const QuaternionT<T>& q1_,
                                     const T& t)
  {
    T cosTheta = dot(q0, q1_);
    QuaternionT<T> q1 = select(cosTheta < 0.f, -q1_, q1_);
    cosTheta          = select(cosTheta < 0.f, -cosTheta, cosTheta);

    // spherical linear interpolation
    const T phi = t * fastapprox::acos(cosTheta);
    T sinPhi, cosPhi;
    fastapprox::sincos(phi, sinPhi, cosPhi);
    QuaternionT<T> qperp = sinPhi * normalize(msub(cosTheta, q0, q1));
    QuaternionT<T> qslerp = msub(cosPhi, q0, qperp);

    // regular linear interpolation as fallback
    QuaternionT<T> qlerp = normalize(lerp(q0, q1, t));

    return select(cosTheta > 0.9995f, qlerp, qslerp);
  }
}
