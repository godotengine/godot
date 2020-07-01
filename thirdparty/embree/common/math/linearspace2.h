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

#include "vec2.h"

namespace embree
{
  ////////////////////////////////////////////////////////////////////////////////
  /// 2D Linear Transform (2x2 Matrix)
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct LinearSpace2
  {
    typedef T Vector;
    typedef typename T::Scalar Scalar;

    /*! default matrix constructor */
    __forceinline LinearSpace2           ( ) {}
    __forceinline LinearSpace2           ( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; }
    __forceinline LinearSpace2& operator=( const LinearSpace2& other ) { vx = other.vx; vy = other.vy; return *this; }

    template<typename L1> __forceinline LinearSpace2( const LinearSpace2<L1>& s ) : vx(s.vx), vy(s.vy) {}

    /*! matrix construction from column vectors */
    __forceinline LinearSpace2(const Vector& vx, const Vector& vy)
      : vx(vx), vy(vy) {}

    /*! matrix construction from row mayor data */
    __forceinline LinearSpace2(const Scalar& m00, const Scalar& m01, 
                               const Scalar& m10, const Scalar& m11)
      : vx(m00,m10), vy(m01,m11) {}

    /*! compute the determinant of the matrix */
    __forceinline const Scalar det() const { return vx.x*vy.y - vx.y*vy.x; }

    /*! compute adjoint matrix */
    __forceinline const LinearSpace2 adjoint() const { return LinearSpace2(vy.y,-vy.x,-vx.y,vx.x); }

    /*! compute inverse matrix */
    __forceinline const LinearSpace2 inverse() const { return adjoint()/det(); }

    /*! compute transposed matrix */
    __forceinline const LinearSpace2 transposed() const { return LinearSpace2(vx.x,vx.y,vy.x,vy.y); }

    /*! returns first row of matrix */
    __forceinline Vector row0() const { return Vector(vx.x,vy.x); }

    /*! returns second row of matrix */
    __forceinline Vector row1() const { return Vector(vx.y,vy.y); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline LinearSpace2( ZeroTy ) : vx(zero), vy(zero) {}
    __forceinline LinearSpace2( OneTy ) : vx(one, zero), vy(zero, one) {}

    /*! return matrix for scaling */
    static __forceinline LinearSpace2 scale(const Vector& s) {
      return LinearSpace2(s.x,   0,
                          0  , s.y);
    }

    /*! return matrix for rotation */
    static __forceinline LinearSpace2 rotate(const Scalar& r) {
      Scalar s = sin(r), c = cos(r);
      return LinearSpace2(c, -s,
                          s,  c);
    }

    /*! return closest orthogonal matrix (i.e. a general rotation including reflection) */
    LinearSpace2 orthogonal() const 
    {
      LinearSpace2 m = *this;

      // mirrored?
      Scalar mirror(one);
      if (m.det() < Scalar(zero)) {
        m.vx = -m.vx;
        mirror = -mirror;
      }

      // rotation
      for (int i = 0; i < 99; i++) {
        const LinearSpace2 m_next = 0.5 * (m + m.transposed().inverse());
        const LinearSpace2 d = m_next - m;
        m = m_next;
        // norm^2 of difference small enough?
        if (max(dot(d.vx, d.vx), dot(d.vy, d.vy)) < 1e-8)
          break;
      }

      // rotation * mirror_x
      return LinearSpace2(mirror*m.vx, m.vy);
    }

  public:

    /*! the column vectors of the matrix */
    Vector vx,vy;
  };

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline LinearSpace2<T> operator -( const LinearSpace2<T>& a ) { return LinearSpace2<T>(-a.vx,-a.vy); }
  template<typename T> __forceinline LinearSpace2<T> operator +( const LinearSpace2<T>& a ) { return LinearSpace2<T>(+a.vx,+a.vy); }
  template<typename T> __forceinline LinearSpace2<T> rcp       ( const LinearSpace2<T>& a ) { return a.inverse(); }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline LinearSpace2<T> operator +( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx+b.vx,a.vy+b.vy); }
  template<typename T> __forceinline LinearSpace2<T> operator -( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return LinearSpace2<T>(a.vx-b.vx,a.vy-b.vy); }

  template<typename T> __forceinline LinearSpace2<T> operator*(const typename T::Scalar & a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }
  template<typename T> __forceinline T               operator*(const LinearSpace2<T>& a, const T              & b) { return b.x*a.vx + b.y*a.vy; }
  template<typename T> __forceinline LinearSpace2<T> operator*(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return LinearSpace2<T>(a*b.vx, a*b.vy); }

  template<typename T> __forceinline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const typename T::Scalar & b) { return LinearSpace2<T>(a.vx/b, a.vy/b); }
  template<typename T> __forceinline LinearSpace2<T> operator/(const LinearSpace2<T>& a, const LinearSpace2<T>& b) { return a * rcp(b); }

  template<typename T> __forceinline LinearSpace2<T>& operator *=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a * b; }
  template<typename T> __forceinline LinearSpace2<T>& operator /=( LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx == b.vx && a.vy == b.vy; }
  template<typename T> __forceinline bool operator !=( const LinearSpace2<T>& a, const LinearSpace2<T>& b ) { return a.vx != b.vx || a.vy != b.vy; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> static std::ostream& operator<<(std::ostream& cout, const LinearSpace2<T>& m) {
    return cout << "{ vx = " << m.vx << ", vy = " << m.vy << "}";
  }

  /*! Shortcuts for common linear spaces. */
  typedef LinearSpace2<Vec2f> LinearSpace2f;
  typedef LinearSpace2<Vec2fa> LinearSpace2fa;
}
