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
  struct Vec3fa;

  ////////////////////////////////////////////////////////////////////////////////
  /// Generic 3D vector Class
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> struct Vec3
  {
    T x, y, z;

    typedef T Scalar;
    enum { N  = 3 };

    ////////////////////////////////////////////////////////////////////////////////
    /// Construction
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3( ) {}
    __forceinline explicit Vec3( const T& a                         ) : x(a), y(a), z(a) {}
    __forceinline          Vec3( const T& x, const T& y, const T& z ) : x(x), y(y), z(z) {}

    __forceinline Vec3( const Vec3& other ) { x = other.x; y = other.y; z = other.z; }
    __forceinline Vec3( const Vec3fa& other );

    template<typename T1> __forceinline Vec3( const Vec3<T1>& a ) : x(T(a.x)), y(T(a.y)), z(T(a.z)) {}
    template<typename T1> __forceinline Vec3& operator =(const Vec3<T1>& other) { x = other.x; y = other.y; z = other.z; return *this; }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline Vec3( ZeroTy   ) : x(zero), y(zero), z(zero) {}
    __forceinline Vec3( OneTy    ) : x(one),  y(one),  z(one) {}
    __forceinline Vec3( PosInfTy ) : x(pos_inf), y(pos_inf), z(pos_inf) {}
    __forceinline Vec3( NegInfTy ) : x(neg_inf), y(neg_inf), z(neg_inf) {}

    __forceinline const T& operator []( const size_t axis ) const { assert(axis < 3); return (&x)[axis]; }
    __forceinline       T& operator []( const size_t axis )       { assert(axis < 3); return (&x)[axis]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T> operator +( const Vec3<T>& a ) { return Vec3<T>(+a.x, +a.y, +a.z); }
  template<typename T> __forceinline Vec3<T> operator -( const Vec3<T>& a ) { return Vec3<T>(-a.x, -a.y, -a.z); }
  template<typename T> __forceinline Vec3<T> abs       ( const Vec3<T>& a ) { return Vec3<T>(abs  (a.x), abs  (a.y), abs  (a.z)); }
  template<typename T> __forceinline Vec3<T> rcp       ( const Vec3<T>& a ) { return Vec3<T>(rcp  (a.x), rcp  (a.y), rcp  (a.z)); }
  template<typename T> __forceinline Vec3<T> rsqrt     ( const Vec3<T>& a ) { return Vec3<T>(rsqrt(a.x), rsqrt(a.y), rsqrt(a.z)); }
  template<typename T> __forceinline Vec3<T> sqrt      ( const Vec3<T>& a ) { return Vec3<T>(sqrt (a.x), sqrt (a.y), sqrt (a.z)); }

  template<typename T> __forceinline Vec3<T> zero_fix( const Vec3<T>& a )
  {
    return Vec3<T>(select(abs(a.x)<min_rcp_input,T(min_rcp_input),a.x),
                   select(abs(a.y)<min_rcp_input,T(min_rcp_input),a.y),
                   select(abs(a.z)<min_rcp_input,T(min_rcp_input),a.z));
  }
  template<typename T> __forceinline Vec3<T> rcp_safe(const Vec3<T>& a) { return rcp(zero_fix(a)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T> operator +( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z); }
  template<typename T> __forceinline Vec3<T> operator -( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z); }
  template<typename T> __forceinline Vec3<T> operator *( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z); }
  template<typename T> __forceinline Vec3<T> operator *( const       T& a, const Vec3<T>& b ) { return Vec3<T>(a   * b.x, a   * b.y, a   * b.z); }
  template<typename T> __forceinline Vec3<T> operator *( const Vec3<T>& a, const       T& b ) { return Vec3<T>(a.x * b  , a.y * b  , a.z * b  ); }
  template<typename T> __forceinline Vec3<T> operator /( const Vec3<T>& a, const       T& b ) { return Vec3<T>(a.x / b  , a.y / b  , a.z / b  ); }
  template<typename T> __forceinline Vec3<T> operator /( const       T& a, const Vec3<T>& b ) { return Vec3<T>(a   / b.x, a   / b.y, a   / b.z); }
  template<typename T> __forceinline Vec3<T> operator /( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z); }

  template<typename T> __forceinline Vec3<T> min(const Vec3<T>& a, const Vec3<T>& b) { return Vec3<T>(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
  template<typename T> __forceinline Vec3<T> max(const Vec3<T>& a, const Vec3<T>& b) { return Vec3<T>(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }

  template<typename T> __forceinline Vec3<T> operator >>( const Vec3<T>& a, const int b ) { return Vec3<T>(a.x >> b, a.y >> b, a.z >> b); }
  template<typename T> __forceinline Vec3<T> operator <<( const Vec3<T>& a, const int b ) { return Vec3<T>(a.x << b, a.y << b, a.z << b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T> madd  ( const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>( madd(a.x,b.x,c.x), madd(a.y,b.y,c.y), madd(a.z,b.z,c.z)); }
  template<typename T> __forceinline Vec3<T> msub  ( const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>( msub(a.x,b.x,c.x), msub(a.y,b.y,c.y), msub(a.z,b.z,c.z)); }
  template<typename T> __forceinline Vec3<T> nmadd ( const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>(nmadd(a.x,b.x,c.x),nmadd(a.y,b.y,c.y),nmadd(a.z,b.z,c.z));}
  template<typename T> __forceinline Vec3<T> nmsub ( const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>(nmsub(a.x,b.x,c.x),nmsub(a.y,b.y,c.y),nmsub(a.z,b.z,c.z)); }

  template<typename T> __forceinline Vec3<T> madd  ( const T& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>( madd(a,b.x,c.x), madd(a,b.y,c.y), madd(a,b.z,c.z)); }
  template<typename T> __forceinline Vec3<T> msub  ( const T& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>( msub(a,b.x,c.x), msub(a,b.y,c.y), msub(a,b.z,c.z)); }
  template<typename T> __forceinline Vec3<T> nmadd ( const T& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>(nmadd(a,b.x,c.x),nmadd(a,b.y,c.y),nmadd(a,b.z,c.z));}
  template<typename T> __forceinline Vec3<T> nmsub ( const T& a, const Vec3<T>& b, const Vec3<T>& c) { return Vec3<T>(nmsub(a,b.x,c.x),nmsub(a,b.y,c.y),nmsub(a,b.z,c.z)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T>& operator +=( Vec3<T>& a, const T        b ) { a.x += b;   a.y += b;   a.z += b;   return a; }
  template<typename T> __forceinline Vec3<T>& operator +=( Vec3<T>& a, const Vec3<T>& b ) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
  template<typename T> __forceinline Vec3<T>& operator -=( Vec3<T>& a, const Vec3<T>& b ) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
  template<typename T> __forceinline Vec3<T>& operator *=( Vec3<T>& a, const       T& b ) { a.x *= b  ; a.y *= b  ; a.z *= b  ; return a; }
  template<typename T> __forceinline Vec3<T>& operator /=( Vec3<T>& a, const       T& b ) { a.x /= b  ; a.y /= b  ; a.z /= b  ; return a; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T reduce_add( const Vec3<T>& a ) { return a.x + a.y + a.z; }
  template<typename T> __forceinline T reduce_mul( const Vec3<T>& a ) { return a.x * a.y * a.z; }
  template<typename T> __forceinline T reduce_min( const Vec3<T>& a ) { return min(a.x, a.y, a.z); }
  template<typename T> __forceinline T reduce_max( const Vec3<T>& a ) { return max(a.x, a.y, a.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline bool operator ==( const Vec3<T>& a, const Vec3<T>& b ) { return a.x == b.x && a.y == b.y && a.z == b.z; }
  template<typename T> __forceinline bool operator !=( const Vec3<T>& a, const Vec3<T>& b ) { return a.x != b.x || a.y != b.y || a.z != b.z; }
  template<typename T> __forceinline bool operator < ( const Vec3<T>& a, const Vec3<T>& b ) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return false;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Shift Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T> shift_right_1( const Vec3<T>& a ) {
    return Vec3<T>(shift_right_1(a.x),shift_right_1(a.y),shift_right_1(a.z));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<T> select ( bool s, const Vec3<T>& t, const Vec3<T>& f ) {
    return Vec3<T>(select(s,t.x,f.x),select(s,t.y,f.y),select(s,t.z,f.z));
  }

  template<typename T> __forceinline Vec3<T> select ( const Vec3<bool>& s, const Vec3<T>& t, const Vec3<T>& f ) {
    return Vec3<T>(select(s.x,t.x,f.x),select(s.y,t.y,f.y),select(s.z,t.z,f.z));
  }

  template<typename T> __forceinline Vec3<T> select ( const typename T::Bool& s, const Vec3<T>& t, const Vec3<T>& f ) {
    return Vec3<T>(select(s,t.x,f.x),select(s,t.y,f.y),select(s,t.z,f.z));
  }

  template<typename T>
    __forceinline Vec3<T> lerp(const Vec3<T>& v0, const Vec3<T>& v1, const T& t) {
    return madd(Vec3<T>(T(1.0f)-t),v0,t*v1);
  }

  template<typename T> __forceinline int maxDim ( const Vec3<T>& a )
  {
    const Vec3<T> b = abs(a);
    if (b.x > b.y) {
      if (b.x > b.z) return 0; else return 2;
    } else {
      if (b.y > b.z) return 1; else return 2;
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline Vec3<bool> eq_mask( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x==b.x,a.y==b.y,a.z==b.z); }
  template<typename T> __forceinline Vec3<bool> neq_mask(const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x!=b.x,a.y!=b.y,a.z!=b.z); }
  template<typename T> __forceinline Vec3<bool> lt_mask( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x< b.x,a.y< b.y,a.z< b.z); }
  template<typename T> __forceinline Vec3<bool> le_mask( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x<=b.x,a.y<=b.y,a.z<=b.z); }
  template<typename T> __forceinline Vec3<bool> gt_mask( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x> b.x,a.y> b.y,a.z> b.z); }
  template<typename T> __forceinline Vec3<bool> ge_mask( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<bool>(a.x>=b.x,a.y>=b.y,a.z>=b.z); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> __forceinline T       dot      ( const Vec3<T>& a, const Vec3<T>& b ) { return madd(a.x,b.x,madd(a.y,b.y,a.z*b.z)); }
  template<typename T> __forceinline T       length   ( const Vec3<T>& a )                   { return sqrt(dot(a,a)); }
  template<typename T> __forceinline T       rcp_length( const Vec3<T>& a )                  { return rsqrt(dot(a,a)); }
  template<typename T> __forceinline Vec3<T> normalize( const Vec3<T>& a )                   { return a*rsqrt(dot(a,a)); }
  template<typename T> __forceinline T       distance ( const Vec3<T>& a, const Vec3<T>& b ) { return length(a-b); }
  template<typename T> __forceinline Vec3<T> cross    ( const Vec3<T>& a, const Vec3<T>& b ) { return Vec3<T>(msub(a.y,b.z,a.z*b.y), msub(a.z,b.x,a.x*b.z), msub(a.x,b.y,a.y*b.x)); }

  template<typename T> __forceinline Vec3<T> stable_triangle_normal( const Vec3<T>& a, const Vec3<T>& b, const Vec3<T>& c )
  {
    const T ab_x = a.z*b.y, ab_y = a.x*b.z, ab_z = a.y*b.x;
    const T bc_x = b.z*c.y, bc_y = b.x*c.z, bc_z = b.y*c.x;
    const Vec3<T> cross_ab(msub(a.y,b.z,ab_x), msub(a.z,b.x,ab_y), msub(a.x,b.y,ab_z));
    const Vec3<T> cross_bc(msub(b.y,c.z,bc_x), msub(b.z,c.x,bc_y), msub(b.x,c.y,bc_z));
    const auto sx = abs(ab_x) < abs(bc_x);
    const auto sy = abs(ab_y) < abs(bc_y);
    const auto sz = abs(ab_z) < abs(bc_z);
    return Vec3<T>(select(sx,cross_ab.x,cross_bc.x),
                   select(sy,cross_ab.y,cross_bc.y),
                   select(sz,cross_ab.z,cross_bc.z));
  }

  template<typename T> __forceinline T       sum      ( const Vec3<T>& a )                   { return a.x+a.y+a.z; }

  template<typename T> __forceinline      T  halfArea ( const Vec3<T>& d )                  { return madd(d.x,(d.y+d.z),d.y*d.z); }
  template<typename T> __forceinline      T  area     ( const Vec3<T>& d )                  { return 2.0f*halfArea(d); }

  template<typename T> __forceinline Vec3<T> normalize_safe( const Vec3<T>& a ) {
    const T d = dot(a,a); return select(d == T( zero ), a ,  a*rsqrt(d) );
  }

  template<typename T> __forceinline T sqr_point_to_line_distance(const Vec3<T>& P, const Vec3<T>& Q0, const Vec3<T>& Q1)
  {
    const Vec3<T> N = cross(P-Q0,Q1-Q0);
    const Vec3<T> D = Q1-Q0;
    return dot(N,N)*rcp(dot(D,D));
  }

  template<typename T> __forceinline T sqr_point_to_line_distance(const Vec3<T>& PmQ0, const Vec3<T>& Q1mQ0)
  {
    const Vec3<T> N = cross(PmQ0,Q1mQ0);
    const Vec3<T> D = Q1mQ0;
    return dot(N,N)*rcp(dot(D,D));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename T> inline std::ostream& operator<<(std::ostream& cout, const Vec3<T>& a) {
    return cout << "(" << a.x << ", " << a.y << ", " << a.z << ")";
  }

  typedef Vec3<bool > Vec3b;
  typedef Vec3<int  > Vec3i;
  typedef Vec3<float> Vec3f;
}

#include "vec3ba.h"
#include "vec3ia.h"
#include "vec3fa.h"

////////////////////////////////////////////////////////////////////////////////
/// SSE / AVX / MIC specializations
////////////////////////////////////////////////////////////////////////////////

#if defined __SSE__
#include "../simd/sse.h"
#endif

#if defined __AVX__
#include "../simd/avx.h"
#endif

#if defined(__AVX512F__)
#include "../simd/avx512.h"
#endif

namespace embree
{
  template<typename Out, typename In>
  __forceinline Vec3<Out> broadcast(const Vec3<In>& a, const size_t k) {
    return Vec3<Out>(Out(a.x[k]), Out(a.y[k]), Out(a.z[k]));
  }

  template<> __forceinline Vec3<float>::Vec3(const Vec3fa& a) { x = a.x; y = a.y; z = a.z; }

#if defined(__AVX__)
  template<> __forceinline Vec3<vfloat4>::Vec3(const Vec3fa& a) {
    x = a.x; y = a.y; z = a.z;
  }
#elif defined(__SSE__)
  template<>
  __forceinline Vec3<vfloat4>::Vec3(const Vec3fa& a) {
    const vfloat4 v = vfloat4(a); x = shuffle<0,0,0,0>(v); y = shuffle<1,1,1,1>(v); z = shuffle<2,2,2,2>(v);
  }
#endif

#if defined(__SSE__)
  __forceinline Vec3<vfloat4> broadcast4f(const Vec3<vfloat4>& a, const size_t k) {
    return Vec3<vfloat4>(vfloat4::broadcast(&a.x[k]), vfloat4::broadcast(&a.y[k]), vfloat4::broadcast(&a.z[k]));
  }

  template<>
  __forceinline Vec3<vfloat4> broadcast<vfloat4,vfloat4>(const Vec3<vfloat4>& a, const size_t k) {
    return Vec3<vfloat4>(vfloat4::broadcast(&a.x[k]), vfloat4::broadcast(&a.y[k]), vfloat4::broadcast(&a.z[k]));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline Vec3<vfloat4> shuffle(const Vec3<vfloat4>& b) {
    return Vec3<vfloat4>(shuffle<i0,i1,i2,i3>(b.x), shuffle<i0,i1,i2,i3>(b.y), shuffle<i0,i1,i2,i3>(b.z));
  }
#endif

#if defined(__AVX__)
  template<>
  __forceinline Vec3<vfloat8>::Vec3(const Vec3fa& a) {
    x = a.x; y = a.y; z = a.z;
  }
  __forceinline Vec3<vfloat4> broadcast4f(const Vec3<vfloat8>& a, const size_t k) {
    return Vec3<vfloat4>(vfloat4::broadcast(&a.x[k]), vfloat4::broadcast(&a.y[k]), vfloat4::broadcast(&a.z[k]));
  }
  __forceinline Vec3<vfloat8> broadcast8f(const Vec3<vfloat4>& a, const size_t k) {
    return Vec3<vfloat8>(vfloat8::broadcast(&a.x[k]), vfloat8::broadcast(&a.y[k]), vfloat8::broadcast(&a.z[k]));
  }
  __forceinline Vec3<vfloat8> broadcast8f(const Vec3<vfloat8>& a, const size_t k) {
    return Vec3<vfloat8>(vfloat8::broadcast(&a.x[k]), vfloat8::broadcast(&a.y[k]), vfloat8::broadcast(&a.z[k]));
  }

  template<>
  __forceinline Vec3<vfloat8> broadcast<vfloat8,vfloat4>(const Vec3<vfloat4>& a, const size_t k) {
    return Vec3<vfloat8>(vfloat8::broadcast(&a.x[k]), vfloat8::broadcast(&a.y[k]), vfloat8::broadcast(&a.z[k]));
  }
  template<>
  __forceinline Vec3<vfloat8> broadcast<vfloat8,vfloat8>(const Vec3<vfloat8>& a, const size_t k) {
    return Vec3<vfloat8>(vfloat8::broadcast(&a.x[k]), vfloat8::broadcast(&a.y[k]), vfloat8::broadcast(&a.z[k]));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline Vec3<vfloat8> shuffle(const Vec3<vfloat8>& b) {
    return Vec3<vfloat8>(shuffle<i0,i1,i2,i3>(b.x), shuffle<i0,i1,i2,i3>(b.y), shuffle<i0,i1,i2,i3>(b.z));
  }
#endif

#if defined(__AVX512F__)
  template<> __forceinline Vec3<vfloat16>::Vec3(const Vec3fa& a) : x(a.x), y(a.y), z(a.z) {}
#endif
}
