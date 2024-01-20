// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Rohit Garg <rpg.314@gmail.com>
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_GEOMETRY_SSE_H
#define EIGEN_GEOMETRY_SSE_H

namespace Eigen { 

namespace internal {

template<class Derived, class OtherDerived>
struct quat_product<Architecture::SSE, Derived, OtherDerived, float>
{
  enum {
    AAlignment = traits<Derived>::Alignment,
    BAlignment = traits<OtherDerived>::Alignment,
    ResAlignment = traits<Quaternion<float> >::Alignment
  };
  static inline Quaternion<float> run(const QuaternionBase<Derived>& _a, const QuaternionBase<OtherDerived>& _b)
  {
    Quaternion<float> res;
    const __m128 mask = _mm_setr_ps(0.f,0.f,0.f,-0.f);
    __m128 a = _a.coeffs().template packet<AAlignment>(0);
    __m128 b = _b.coeffs().template packet<BAlignment>(0);
    __m128 s1 = _mm_mul_ps(vec4f_swizzle1(a,1,2,0,2),vec4f_swizzle1(b,2,0,1,2));
    __m128 s2 = _mm_mul_ps(vec4f_swizzle1(a,3,3,3,1),vec4f_swizzle1(b,0,1,2,1));
    pstoret<float,Packet4f,ResAlignment>(
              &res.x(),
              _mm_add_ps(_mm_sub_ps(_mm_mul_ps(a,vec4f_swizzle1(b,3,3,3,3)),
                                    _mm_mul_ps(vec4f_swizzle1(a,2,0,1,0),
                                               vec4f_swizzle1(b,1,2,0,0))),
                         _mm_xor_ps(mask,_mm_add_ps(s1,s2))));
    
    return res;
  }
};

template<class Derived>
struct quat_conj<Architecture::SSE, Derived, float>
{
  enum {
    ResAlignment = traits<Quaternion<float> >::Alignment
  };
  static inline Quaternion<float> run(const QuaternionBase<Derived>& q)
  {
    Quaternion<float> res;
    const __m128 mask = _mm_setr_ps(-0.f,-0.f,-0.f,0.f);
    pstoret<float,Packet4f,ResAlignment>(&res.x(), _mm_xor_ps(mask, q.coeffs().template packet<traits<Derived>::Alignment>(0)));
    return res;
  }
};


template<typename VectorLhs,typename VectorRhs>
struct cross3_impl<Architecture::SSE,VectorLhs,VectorRhs,float,true>
{
  enum {
    ResAlignment = traits<typename plain_matrix_type<VectorLhs>::type>::Alignment
  };
  static inline typename plain_matrix_type<VectorLhs>::type
  run(const VectorLhs& lhs, const VectorRhs& rhs)
  {
    __m128 a = lhs.template packet<traits<VectorLhs>::Alignment>(0);
    __m128 b = rhs.template packet<traits<VectorRhs>::Alignment>(0);
    __m128 mul1=_mm_mul_ps(vec4f_swizzle1(a,1,2,0,3),vec4f_swizzle1(b,2,0,1,3));
    __m128 mul2=_mm_mul_ps(vec4f_swizzle1(a,2,0,1,3),vec4f_swizzle1(b,1,2,0,3));
    typename plain_matrix_type<VectorLhs>::type res;
    pstoret<float,Packet4f,ResAlignment>(&res.x(),_mm_sub_ps(mul1,mul2));
    return res;
  }
};




template<class Derived, class OtherDerived>
struct quat_product<Architecture::SSE, Derived, OtherDerived, double>
{
  enum {
    BAlignment = traits<OtherDerived>::Alignment,
    ResAlignment = traits<Quaternion<double> >::Alignment
  };

  static inline Quaternion<double> run(const QuaternionBase<Derived>& _a, const QuaternionBase<OtherDerived>& _b)
  {
  const Packet2d mask = _mm_castsi128_pd(_mm_set_epi32(0x0,0x0,0x80000000,0x0));

  Quaternion<double> res;

  const double* a = _a.coeffs().data();
  Packet2d b_xy = _b.coeffs().template packet<BAlignment>(0);
  Packet2d b_zw = _b.coeffs().template packet<BAlignment>(2);
  Packet2d a_xx = pset1<Packet2d>(a[0]);
  Packet2d a_yy = pset1<Packet2d>(a[1]);
  Packet2d a_zz = pset1<Packet2d>(a[2]);
  Packet2d a_ww = pset1<Packet2d>(a[3]);

  // two temporaries:
  Packet2d t1, t2;

  /*
   * t1 = ww*xy + yy*zw
   * t2 = zz*xy - xx*zw
   * res.xy = t1 +/- swap(t2)
   */
  t1 = padd(pmul(a_ww, b_xy), pmul(a_yy, b_zw));
  t2 = psub(pmul(a_zz, b_xy), pmul(a_xx, b_zw));
#ifdef EIGEN_VECTORIZE_SSE3
  EIGEN_UNUSED_VARIABLE(mask)
  pstoret<double,Packet2d,ResAlignment>(&res.x(), _mm_addsub_pd(t1, preverse(t2)));
#else
  pstoret<double,Packet2d,ResAlignment>(&res.x(), padd(t1, pxor(mask,preverse(t2))));
#endif
  
  /*
   * t1 = ww*zw - yy*xy
   * t2 = zz*zw + xx*xy
   * res.zw = t1 -/+ swap(t2) = swap( swap(t1) +/- t2)
   */
  t1 = psub(pmul(a_ww, b_zw), pmul(a_yy, b_xy));
  t2 = padd(pmul(a_zz, b_zw), pmul(a_xx, b_xy));
#ifdef EIGEN_VECTORIZE_SSE3
  EIGEN_UNUSED_VARIABLE(mask)
  pstoret<double,Packet2d,ResAlignment>(&res.z(), preverse(_mm_addsub_pd(preverse(t1), t2)));
#else
  pstoret<double,Packet2d,ResAlignment>(&res.z(), psub(t1, pxor(mask,preverse(t2))));
#endif

  return res;
}
};

template<class Derived>
struct quat_conj<Architecture::SSE, Derived, double>
{
  enum {
    ResAlignment = traits<Quaternion<double> >::Alignment
  };
  static inline Quaternion<double> run(const QuaternionBase<Derived>& q)
  {
    Quaternion<double> res;
    const __m128d mask0 = _mm_setr_pd(-0.,-0.);
    const __m128d mask2 = _mm_setr_pd(-0.,0.);
    pstoret<double,Packet2d,ResAlignment>(&res.x(), _mm_xor_pd(mask0, q.coeffs().template packet<traits<Derived>::Alignment>(0)));
    pstoret<double,Packet2d,ResAlignment>(&res.z(), _mm_xor_pd(mask2, q.coeffs().template packet<traits<Derived>::Alignment>(2)));
    return res;
  }
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_GEOMETRY_SSE_H
