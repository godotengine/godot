
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_ARCH_CONJ_HELPER_H
#define EIGEN_ARCH_CONJ_HELPER_H

#define EIGEN_MAKE_CONJ_HELPER_CPLX_REAL(PACKET_CPLX, PACKET_REAL)                                                          \
  template<> struct conj_helper<PACKET_REAL, PACKET_CPLX, false,false> {                                          \
    EIGEN_STRONG_INLINE PACKET_CPLX pmadd(const PACKET_REAL& x, const PACKET_CPLX& y, const PACKET_CPLX& c) const \
    { return padd(c, pmul(x,y)); }                                                                                \
    EIGEN_STRONG_INLINE PACKET_CPLX pmul(const PACKET_REAL& x, const PACKET_CPLX& y) const                        \
    { return PACKET_CPLX(Eigen::internal::pmul<PACKET_REAL>(x, y.v)); }                                           \
  };                                                                                                              \
                                                                                                                  \
  template<> struct conj_helper<PACKET_CPLX, PACKET_REAL, false,false> {                                          \
    EIGEN_STRONG_INLINE PACKET_CPLX pmadd(const PACKET_CPLX& x, const PACKET_REAL& y, const PACKET_CPLX& c) const \
    { return padd(c, pmul(x,y)); }                                                                                \
    EIGEN_STRONG_INLINE PACKET_CPLX pmul(const PACKET_CPLX& x, const PACKET_REAL& y) const                        \
    { return PACKET_CPLX(Eigen::internal::pmul<PACKET_REAL>(x.v, y)); }                                           \
  };

#endif // EIGEN_ARCH_CONJ_HELPER_H
