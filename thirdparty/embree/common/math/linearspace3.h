// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec3.h"
#include "quaternion.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// 3D Linear Transform (3x3 Matrix)
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct LinearSpace3
  {
    typedef T Vector;
    typedef typename T::Scalar Scalar;

    /*! default matrix constructor */
    __forceinline LinearSpace3           ( ) {}
    __forceinline LinearSpace3           ( const LinearSpace3& other ) { vx = other.vx; vy = other.vy; vz = other.vz; }
    __forceinline LinearSpace3& operator=( const LinearSpace3& other ) { vx = other.vx; vy = other.vy; vz = other.vz; return *this; }

    template<typename L1> __forceinline LinearSpace3( const LinearSpace3<L1>& s ) : vx(s.vx), vy(s.vy), vz(s.vz) {}

    /*! matrix construction from column vectors */
    __forceinline LinearSpace3(const Vector& vx, const Vector& vy, const Vector& vz)
      : vx(vx), vy(vy), vz(vz) {}

    /*! construction from quaternion */
    __forceinline LinearSpace3( const QuaternionT<Scalar>& q )
      : vx((q.r*q.r + q.i*q.i - q.j*q.j - q.k*q.k), 2.0f*(q.i*q.j + q.r*q.k), 2.0f*(q.i*q.k - q.r*q.j))
      , vy(2.0f*(q.i*q.j - q.r*q.k), (q.r*q.r - q.i*q.i + q.j*q.j - q.k*q.k), 2.0f*(q.j*q.k + q.r*q.i))
      , vz(2.0f*(q.i*q.k + q.r*q.j), 2.0f*(q.j*q.k - q.r*q.i), (q.r*q.r - q.i*q.i - q.j*q.j + q.k*q.k)) {}

    /*! matrix construction from row mayor data */
    __forceinline LinearSpace3(const Scalar& m00, const Scalar& m01, const Scalar& m02,
                               const Scalar& m10, const Scalar& m11, const Scalar& m12,
                               const Scalar& m20, const Scalar& m21, const Scalar& m22)
      : vx(m00,m10,m20), vy(m01,m11,m21), vz(m02,m12,m22) {}

    /*! compute the determinant of the matrix */
    __forceinline const Scalar det() const { return dot(vx,cross(vy,vz)); }

    /*! compute adjoint matrix */
    __forceinline const LinearSpace3 adjoint() const { return LinearSpace3(cross(vy,vz),cross(vz,vx),cross(vx,vy)).transposed(); }

    /*! compute inverse matrix */
    __forceinline const LinearSpace3 inverse() const { return adjoint()/det(); }

    /*! compute transposed matrix */
    __forceinline const LinearSpace3 transposed() const { return LinearSpace3(vx.x,vx.y,vx.z,vy.x,vy.y,vy.z,vz.x,vz.y,vz.z); }

    /*! returns first row of matrix */
    __forceinline Vector row0() const { return Vector(vx.x,vy.x,vz.x); }

    /*! returns second row of matrix */
    __forceinline Vector row1() const { return Vector(vx.y,vy.y,vz.y); }

    /*! returns third row of matrix */
    __forceinline Vector row2() const { return Vector(vx.z,vy.z,vz.z); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline LinearSpace3( ZeroTy ) : vx(zero), vy(zero), vz(zero) {}
    __forceinline LinearSpace3( OneTy ) : vx(one, zero, zero), vy(zero, one, zero), vz(zero, zero, one) {}

    /*! return matrix for scaling */
    static __forceinline LinearSpace3 scale(const Vector& s) {
      return LinearSpace3(s.x,   0,   0,
                          0  , s.y,   0,
                          0  ,   0, s.z);
    }

    /*! return matrix for rotation around arbitrary axis */
    static __forceinline LinearSpace3 rotate(const Vector& _u, const Scalar& r) {
      Vector u = normalize(_u);
      Scalar s = sin(r), c = cos(r);
      return LinearSpace3(u.x*u.x+(1-u.x*u.x)*c,  u.x*u.y*(1-c)-u.z*s,    u.x*u.z*(1-c)+u.y*s,
                          u.x*u.y*(1-c)+u.z*s,    u.y*u.y+(1-u.y*u.y)*c,  u.y*u.z*(1-c)-u.x*s,
                          u.x*u.z*(1-c)-u.y*s,    u.y*u.z*(1-c)+u.x*s,    u.z*u.z+(1-u.z*u.z)*c);
    }

  public:

    /*! the column vectors of the matrix */
    Vector vx,vy,vz;
  };

  /*! compute transposed matrix */
  template<> __forceinline const LinearSpace3<Vec3fa> LinearSpace3<Vec3fa>::transposed() const { 
    vfloat4 rx,ry,rz; transpose((vfloat4&)vx,(vfloat4&)vy,(vfloat4&)vz,vfloat4(zero),rx,ry,rz);
    return LinearSpace3<Vec3fa>(Vec3fa(rx),Vec3fa(ry),Vec3fa(rz)); 
  }

  template<typename T>
    __forceinline const LinearSpace3<T> transposed(const LinearSpace3<T>& xfm) { 
    return xfm.transposed();
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline LinearSpace3<T> operator -( const LinearSpace3<T>& a ) { return LinearSpace3<T>(-a.vx,-a.vy,-a.vz); }
  template<typename T> __forceinline LinearSpace3<T> operator +( const LinearSpace3<T>& a ) { return LinearSpace3<T>(+a.vx,+a.vy,+a.vz); }
  template<typename T> __forceinline LinearSpace3<T> rcp       ( const LinearSpace3<T>& a ) { return a.inverse(); }

  /* constructs a coordinate frame form a normalized normal */
  template<typename T> __forceinline LinearSpace3<T> frame(const T& N) 
  {
    const T dx0(0,N.z,-N.y);
    const T dx1(-N.z,0,N.x);
    const T dx = normalize(select(dot(dx0,dx0) > dot(dx1,dx1),dx0,dx1));
    const T dy = normalize(cross(N,dx));
    return LinearSpace3<T>(dx,dy,N);
  }

  /* constructs a coordinate frame from a normal and approximate x-direction */
  template<typename T> __forceinline LinearSpace3<T> frame(const T& N, const T& dxi)
  {
    if (abs(dot(dxi,N)) > 0.99f) return frame(N); // fallback in case N and dxi are very parallel
    const T dx = normalize(cross(dxi,N));
    const T dy = normalize(cross(N,dx));
    return LinearSpace3<T>(dx,dy,N);
  }
  
  /* clamps linear space to range -1 to +1 */
  template<typename T> __forceinline LinearSpace3<T> clamp(const LinearSpace3<T>& space) {
    return LinearSpace3<T>(clamp(space.vx,T(-1.0f),T(1.0f)),
                           clamp(space.vy,T(-1.0f),T(1.0f)),
                           clamp(space.vz,T(-1.0f),T(1.0f)));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline LinearSpace3<T> operator +( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return LinearSpace3<T>(a.vx+b.vx,a.vy+b.vy,a.vz+b.vz); }
  template<typename T> __forceinline LinearSpace3<T> operator -( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return LinearSpace3<T>(a.vx-b.vx,a.vy-b.vy,a.vz-b.vz); }

  template<typename T> __forceinline LinearSpace3<T> operator*(const typename T::Scalar & a, const LinearSpace3<T>& b) { return LinearSpace3<T>(a*b.vx, a*b.vy, a*b.vz); }
  template<typename T> __forceinline T               operator*(const LinearSpace3<T>& a, const T              & b) { return madd(T(b.x),a.vx,madd(T(b.y),a.vy,T(b.z)*a.vz)); }
  template<typename T> __forceinline LinearSpace3<T> operator*(const LinearSpace3<T>& a, const LinearSpace3<T>& b) { return LinearSpace3<T>(a*b.vx, a*b.vy, a*b.vz); }

  template<typename T> __forceinline LinearSpace3<T> operator/(const LinearSpace3<T>& a, const typename T::Scalar & b) { return LinearSpace3<T>(a.vx/b, a.vy/b, a.vz/b); }
  template<typename T> __forceinline LinearSpace3<T> operator/(const LinearSpace3<T>& a, const LinearSpace3<T>& b) { return a * rcp(b); }

  template<typename T> __forceinline LinearSpace3<T>& operator *=( LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a = a * b; }
  template<typename T> __forceinline LinearSpace3<T>& operator /=( LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a = a / b; }

  template<typename T> __forceinline T       xfmPoint (const LinearSpace3<T>& s, const T      & a) { return madd(T(a.x),s.vx,madd(T(a.y),s.vy,T(a.z)*s.vz)); }
  template<typename T> __forceinline T       xfmVector(const LinearSpace3<T>& s, const T      & a) { return madd(T(a.x),s.vx,madd(T(a.y),s.vy,T(a.z)*s.vz)); }
  template<typename T> __forceinline T       xfmNormal(const LinearSpace3<T>& s, const T      & a) { return xfmVector(s.inverse().transposed(),a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a.vx == b.vx && a.vy == b.vy && a.vz == b.vz; }
  template<typename T> __forceinline bool operator !=( const LinearSpace3<T>& a, const LinearSpace3<T>& b ) { return a.vx != b.vx || a.vy != b.vy || a.vz != b.vz; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline LinearSpace3<T> select ( const typename T::Scalar::Bool& s, const LinearSpace3<T>& t, const LinearSpace3<T>& f ) {
    return LinearSpace3<T>(select(s,t.vx,f.vx),select(s,t.vy,f.vy),select(s,t.vz,f.vz));
  }

  /*! blending */
  template<typename T>
    __forceinline LinearSpace3<T> lerp(const LinearSpace3<T>& l0, const LinearSpace3<T>& l1, const float t) 
  {
    return LinearSpace3<T>(lerp(l0.vx,l1.vx,t),
                           lerp(l0.vy,l1.vy,t),
                           lerp(l0.vz,l1.vz,t));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> static embree_ostream operator<<(embree_ostream cout, const LinearSpace3<T>& m) {
    return cout << "{ vx = " << m.vx << ", vy = " << m.vy << ", vz = " << m.vz << "}";
  }

  /*! Shortcuts for common linear spaces. */
  typedef LinearSpace3<Vec3f> LinearSpace3f;
  typedef LinearSpace3<Vec3fa> LinearSpace3fa;
  typedef LinearSpace3<Vec3fx> LinearSpace3fx;
  typedef LinearSpace3<Vec3ff> LinearSpace3ff;

  template<int N> using LinearSpace3vf = LinearSpace3<Vec3<vfloat<N>>>;
  typedef LinearSpace3<Vec3<vfloat<4>>>  LinearSpace3vf4;
  typedef LinearSpace3<Vec3<vfloat<8>>>  LinearSpace3vf8;
  typedef LinearSpace3<Vec3<vfloat<16>>> LinearSpace3vf16;

  /*! blending */
  template<typename T, typename S>
    __forceinline LinearSpace3<T> lerp(const LinearSpace3<T>& l0,
                                       const LinearSpace3<T>& l1,
                                       const S& t)
  {
    return LinearSpace3<T>(lerp(l0.vx,l1.vx,t),
                           lerp(l0.vy,l1.vy,t),
                           lerp(l0.vz,l1.vz,t));
  }

}
