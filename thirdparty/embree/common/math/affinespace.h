// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "linearspace2.h"
#include "linearspace3.h"
#include "quaternion.h"
#include "bbox.h"
#include "vec4.h"

namespace embree
{
  #define VectorT typename L::Vector
  #define ScalarT typename L::Vector::Scalar

  ////////////////////////////////////////////////////////////////////////////////
  // Affine Space
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L>
    struct AffineSpaceT
    {
      L l;           /*< linear part of affine space */
      VectorT p;     /*< affine part of affine space */

      ////////////////////////////////////////////////////////////////////////////////
      // Constructors, Assignment, Cast, Copy Operations
      ////////////////////////////////////////////////////////////////////////////////

      __forceinline AffineSpaceT           ( )                           { }
      __forceinline AffineSpaceT           ( const AffineSpaceT& other ) { l = other.l; p = other.p; }
      __forceinline AffineSpaceT           ( const L           & other ) { l = other  ; p = VectorT(zero); }
      __forceinline AffineSpaceT& operator=( const AffineSpaceT& other ) { l = other.l; p = other.p; return *this; }

      __forceinline AffineSpaceT( const VectorT& vx, const VectorT& vy, const VectorT& vz, const VectorT& p ) : l(vx,vy,vz), p(p) {}
      __forceinline AffineSpaceT( const L& l, const VectorT& p ) : l(l), p(p) {}

      template<typename L1> __forceinline AffineSpaceT( const AffineSpaceT<L1>& s ) : l(s.l), p(s.p) {}

      ////////////////////////////////////////////////////////////////////////////////
      // Constants
      ////////////////////////////////////////////////////////////////////////////////

      __forceinline AffineSpaceT( ZeroTy ) : l(zero), p(zero) {}
      __forceinline AffineSpaceT( OneTy )  : l(one),  p(zero) {}

      /*! return matrix for scaling */
      static __forceinline AffineSpaceT scale(const VectorT& s) { return L::scale(s); }

      /*! return matrix for translation */
      static __forceinline AffineSpaceT translate(const VectorT& p) { return AffineSpaceT(one,p); }

      /*! return matrix for rotation, only in 2D */
      static __forceinline AffineSpaceT rotate(const ScalarT& r) { return L::rotate(r); }

      /*! return matrix for rotation around arbitrary point (2D) or axis (3D) */
      static __forceinline AffineSpaceT rotate(const VectorT& u, const ScalarT& r) { return L::rotate(u,r); }

      /*! return matrix for rotation around arbitrary axis and point, only in 3D */
      static __forceinline AffineSpaceT rotate(const VectorT& p, const VectorT& u, const ScalarT& r) { return translate(+p) * rotate(u,r) * translate(-p);  }

      /*! return matrix for looking at given point, only in 3D */
      static __forceinline AffineSpaceT lookat(const VectorT& eye, const VectorT& point, const VectorT& up) {
        VectorT Z = normalize(point-eye);
        VectorT U = normalize(cross(up,Z));
        VectorT V = normalize(cross(Z,U));
        return AffineSpaceT(L(U,V,Z),eye);
      }

    };
  
  // template specialization to get correct identity matrix for type AffineSpace3fa
  template<>
    __forceinline AffineSpaceT<LinearSpace3ff>::AffineSpaceT( OneTy )  : l(one),  p(0.f, 0.f, 0.f, 1.f) {}

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> __forceinline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(-a.l,-a.p); }
  template<typename L> __forceinline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(+a.l,+a.p); }
  template<typename L> __forceinline AffineSpaceT<L>        rcp( const AffineSpaceT<L>& a ) { L il = rcp(a.l); return AffineSpaceT<L>(il,-(il*a.p)); }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> __forceinline const AffineSpaceT<L> operator +( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l+b.l,a.p+b.p); }
  template<typename L> __forceinline const AffineSpaceT<L> operator -( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l-b.l,a.p-b.p); }

  template<typename L> __forceinline const AffineSpaceT<L> operator *( const ScalarT        & a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a*b.l,a*b.p); }
  template<typename L> __forceinline const AffineSpaceT<L> operator *( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l*b.l,a.l*b.p+a.p); }
  template<typename L> __forceinline const AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a * rcp(b); }
  template<typename L> __forceinline const AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const ScalarT        & b ) { return a * rcp(b); }

  template<typename L> __forceinline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a * b; }
  template<typename L> __forceinline AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a * b; }
  template<typename L> __forceinline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a = a / b; }
  template<typename L> __forceinline AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const ScalarT        & b ) { return a = a / b; }

  template<typename L> __forceinline VectorT xfmPoint (const AffineSpaceT<L>& m, const VectorT& p) { return madd(VectorT(p.x),m.l.vx,madd(VectorT(p.y),m.l.vy,madd(VectorT(p.z),m.l.vz,m.p))); }
  template<typename L> __forceinline VectorT xfmVector(const AffineSpaceT<L>& m, const VectorT& v) { return xfmVector(m.l,v); }
  template<typename L> __forceinline VectorT xfmNormal(const AffineSpaceT<L>& m, const VectorT& n) { return xfmNormal(m.l,n); }

  __forceinline const BBox<Vec3fa> xfmBounds(const AffineSpaceT<LinearSpace3<Vec3fa> >& m, const BBox<Vec3fa>& b) 
  { 
    BBox3fa dst = empty;
    const Vec3fa p0(b.lower.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p0));
    const Vec3fa p1(b.lower.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p1));
    const Vec3fa p2(b.lower.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p2));
    const Vec3fa p3(b.lower.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p3));
    const Vec3fa p4(b.upper.x,b.lower.y,b.lower.z); dst.extend(xfmPoint(m,p4));
    const Vec3fa p5(b.upper.x,b.lower.y,b.upper.z); dst.extend(xfmPoint(m,p5));
    const Vec3fa p6(b.upper.x,b.upper.y,b.lower.z); dst.extend(xfmPoint(m,p6));
    const Vec3fa p7(b.upper.x,b.upper.y,b.upper.z); dst.extend(xfmPoint(m,p7));
    return dst;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> __forceinline bool operator ==( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l == b.l && a.p == b.p; }
  template<typename L> __forceinline bool operator !=( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return a.l != b.l || a.p != b.p; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> __forceinline AffineSpaceT<L> select ( const typename L::Vector::Scalar::Bool& s, const AffineSpaceT<L>& t, const AffineSpaceT<L>& f ) {
    return AffineSpaceT<L>(select(s,t.l,f.l),select(s,t.p,f.p));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> static embree_ostream operator<<(embree_ostream cout, const AffineSpaceT<L>& m) {
    return cout << "{ l = " << m.l << ", p = " << m.p << " }";
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template Instantiations
  ////////////////////////////////////////////////////////////////////////////////

  typedef AffineSpaceT<LinearSpace2f> AffineSpace2f;
  typedef AffineSpaceT<LinearSpace3f> AffineSpace3f;
  typedef AffineSpaceT<LinearSpace3fa> AffineSpace3fa;
  typedef AffineSpaceT<LinearSpace3fx> AffineSpace3fx;
  typedef AffineSpaceT<LinearSpace3ff> AffineSpace3ff;
  typedef AffineSpaceT<Quaternion3f > OrthonormalSpace3f;

  template<int N> using AffineSpace3vf = AffineSpaceT<LinearSpace3<Vec3<vfloat<N>>>>;
  typedef AffineSpaceT<LinearSpace3<Vec3<vfloat<4>>>>  AffineSpace3vf4;
  typedef AffineSpaceT<LinearSpace3<Vec3<vfloat<8>>>>  AffineSpace3vf8;
  typedef AffineSpaceT<LinearSpace3<Vec3<vfloat<16>>>> AffineSpace3vf16;

  template<int N> using AffineSpace3vff = AffineSpaceT<LinearSpace3<Vec4<vfloat<N>>>>;
  typedef AffineSpaceT<LinearSpace3<Vec4<vfloat<4>>>>  AffineSpace3vfa4;
  typedef AffineSpaceT<LinearSpace3<Vec4<vfloat<8>>>>  AffineSpace3vfa8;
  typedef AffineSpaceT<LinearSpace3<Vec4<vfloat<16>>>> AffineSpace3vfa16;

  //////////////////////////////////////////////////////////////////////////////
  /// Interpolation
  //////////////////////////////////////////////////////////////////////////////
  template<typename T, typename R>
  __forceinline AffineSpaceT<T> lerp(const AffineSpaceT<T>& M0,
                                     const AffineSpaceT<T>& M1,
                                     const R& t)
  {
    return AffineSpaceT<T>(lerp(M0.l,M1.l,t),lerp(M0.p,M1.p,t));
  }

  // slerp interprets the 16 floats of the matrix M = D * R * S as components of
  // three matrizes (D, R, S) that are interpolated individually.
  template<typename T> __forceinline AffineSpaceT<LinearSpace3<Vec3<T>>>
  slerp(const AffineSpaceT<LinearSpace3<Vec4<T>>>& M0,
        const AffineSpaceT<LinearSpace3<Vec4<T>>>& M1,
        const T& t)
  {
    QuaternionT<T> q0(M0.p.w, M0.l.vx.w, M0.l.vy.w, M0.l.vz.w);
    QuaternionT<T> q1(M1.p.w, M1.l.vx.w, M1.l.vy.w, M1.l.vz.w);
    QuaternionT<T> q = slerp(q0, q1, t);

    AffineSpaceT<LinearSpace3<Vec3<T>>> S = lerp(M0, M1, t);
    AffineSpaceT<LinearSpace3<Vec3<T>>> D(one);
    D.p.x = S.l.vx.y;
    D.p.y = S.l.vx.z;
    D.p.z = S.l.vy.z;
    S.l.vx.y = 0;
    S.l.vx.z = 0;
    S.l.vy.z = 0;

    AffineSpaceT<LinearSpace3<Vec3<T>>> R = LinearSpace3<Vec3<T>>(q);
    return D * R * S;
  }

  // this is a specialized version for Vec3fa because that does
  // not play along nicely with the other templated Vec3/Vec4 types
  __forceinline AffineSpace3fa slerp(const AffineSpace3ff& M0,
                                     const AffineSpace3ff& M1,
                                     const float& t)
  {
    Quaternion3f q0(M0.p.w, M0.l.vx.w, M0.l.vy.w, M0.l.vz.w);
    Quaternion3f q1(M1.p.w, M1.l.vx.w, M1.l.vy.w, M1.l.vz.w);
    Quaternion3f q = slerp(q0, q1, t);

    AffineSpace3fa S = lerp(M0, M1, t);
    AffineSpace3fa D(one);
    D.p.x = S.l.vx.y;
    D.p.y = S.l.vx.z;
    D.p.z = S.l.vy.z;
    S.l.vx.y = 0;
    S.l.vx.z = 0;
    S.l.vy.z = 0;

    AffineSpace3fa R = LinearSpace3fa(q);
    return D * R * S;
  }
  
  __forceinline AffineSpace3fa quaternionDecompositionToAffineSpace(const AffineSpace3ff& qd)
  {
    // compute affine transform from quaternion decomposition
    Quaternion3f q(qd.p.w, qd.l.vx.w, qd.l.vy.w, qd.l.vz.w);
    AffineSpace3fa M = qd;
    AffineSpace3fa D(one);
    D.p.x = M.l.vx.y;
    D.p.y = M.l.vx.z;
    D.p.z = M.l.vy.z;
    M.l.vx.y = 0;
    M.l.vx.z = 0;
    M.l.vy.z = 0;
    AffineSpace3fa R = LinearSpace3fa(q);
    return D * R * M;
  }
  
  __forceinline void quaternionDecomposition(const AffineSpace3ff& qd, Vec3fa& T, Quaternion3f& q, AffineSpace3fa& S)
  {
    q = Quaternion3f(qd.p.w, qd.l.vx.w, qd.l.vy.w, qd.l.vz.w);
    S = qd;
    T.x = qd.l.vx.y;
    T.y = qd.l.vx.z;
    T.z = qd.l.vy.z;
    S.l.vx.y = 0;
    S.l.vx.z = 0;
    S.l.vy.z = 0;
  }

  __forceinline AffineSpace3fx quaternionDecomposition(Vec3fa const& T, Quaternion3f const& q, AffineSpace3fa const& S)
  {
    AffineSpace3ff M = S;
    M.l.vx.w = q.i;
    M.l.vy.w = q.j;
    M.l.vz.w = q.k;
    M.p.w    = q.r;
    M.l.vx.y = T.x;
    M.l.vx.z = T.y;
    M.l.vy.z = T.z;
    return M;
  }

  struct __aligned(16) QuaternionDecomposition
  {
    float scale_x = 1.f;
    float scale_y = 1.f;
    float scale_z = 1.f;
    float skew_xy = 0.f;
    float skew_xz = 0.f;
    float skew_yz = 0.f;
    float shift_x = 0.f;
    float shift_y = 0.f;
    float shift_z = 0.f;
    float quaternion_r = 1.f;
    float quaternion_i = 0.f;
    float quaternion_j = 0.f;
    float quaternion_k = 0.f;
    float translation_x = 0.f;
    float translation_y = 0.f;
    float translation_z = 0.f;
  };

  __forceinline QuaternionDecomposition quaternionDecomposition(AffineSpace3ff const& M)
  {
    QuaternionDecomposition qd;
    qd.scale_x       = M.l.vx.x;
    qd.scale_y       = M.l.vy.y;
    qd.scale_z       = M.l.vz.z;
    qd.shift_x       = M.p.x;
    qd.shift_y       = M.p.y;
    qd.shift_z       = M.p.z;
    qd.translation_x = M.l.vx.y;
    qd.translation_y = M.l.vx.z;
    qd.translation_z = M.l.vy.z;
    qd.skew_xy       = M.l.vy.x;
    qd.skew_xz       = M.l.vz.x;
    qd.skew_yz       = M.l.vz.y;
    qd.quaternion_r  = M.p.w;
    qd.quaternion_i  = M.l.vx.w;
    qd.quaternion_j  = M.l.vy.w;
    qd.quaternion_k  = M.l.vz.w;
    return qd;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /*
   * ! Template Specialization for 2D: return matrix for rotation around point
   * (rotation around arbitrarty vector is not meaningful in 2D)
   */
  template<> __forceinline
  AffineSpace2f AffineSpace2f::rotate(const Vec2f& p, const float& r) {
    return translate(+p)*AffineSpace2f(LinearSpace2f::rotate(r))*translate(-p);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Similarity Transform
  //
  // checks, if M is a similarity transformation, i.e if there exists a factor D
  // such that for all x,y: distance(Mx, My) = D * distance(x, y)
  ////////////////////////////////////////////////////////////////////////////////
  __forceinline bool similarityTransform(const AffineSpace3fa& M, float* D)
  {
    if (D) *D = 0.f;
    if (abs(dot(M.l.vx, M.l.vy)) > 1e-5f) return false;
    if (abs(dot(M.l.vx, M.l.vz)) > 1e-5f) return false;
    if (abs(dot(M.l.vy, M.l.vz)) > 1e-5f) return false;

    const float D_x = dot(M.l.vx, M.l.vx);
    const float D_y = dot(M.l.vy, M.l.vy);
    const float D_z = dot(M.l.vz, M.l.vz);

    if (abs(D_x - D_y) > 1e-5f ||
        abs(D_x - D_z) > 1e-5f ||
        abs(D_y - D_z) > 1e-5f)
      return false;

    if (D) *D = sqrtf(D_x);
    return true;
  }

  __forceinline void AffineSpace3fa_store_unaligned(const AffineSpace3fa &source, AffineSpace3fa* ptr)
  {
    Vec3fa::storeu(&ptr->l.vx, source.l.vx);
    Vec3fa::storeu(&ptr->l.vy, source.l.vy);
    Vec3fa::storeu(&ptr->l.vz, source.l.vz);
    Vec3fa::storeu(&ptr->p, source.p);
  }

  __forceinline AffineSpace3fa AffineSpace3fa_load_unaligned(AffineSpace3fa* ptr)
  {
    AffineSpace3fa space;
    space.l.vx = Vec3fa::loadu(&ptr->l.vx);
    space.l.vy = Vec3fa::loadu(&ptr->l.vy);
    space.l.vz = Vec3fa::loadu(&ptr->l.vz);
    space.p    = Vec3fa::loadu(&ptr->p);
    return space;
  }

  #undef VectorT
  #undef ScalarT
}
