/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef OS_BLAS_HPP
#define OS_BLAS_HPP

/** \file
 * Common stuff respecting USE_MKL and USE_CBLAS compile flags
 *
 *  USE_MKL  USE_CBLAS effect
 *  -------  --------- ------
 *  yes      yes       normal compile: jit *may* be preferred over Intel(R) MKL CBLAS
 *  yes      no        jit calls OK; assert if cblas is ever called
 *  no       yes       system-dependent CBLAS
 *  no       no        gemm convolution (or other blas) N/A; create stubs
 */

#if defined(USE_MKL)

#include "mkl_version.h"

#define USE_MKL_PACKED_GEMM (INTEL_MKL_VERSION >= 20190001)
#define USE_MKL_IGEMM \
    (INTEL_MKL_VERSION >= 20180000 && __INTEL_MKL_BUILD_DATE >= 20170628)

#include "mkl_cblas.h"
#if !defined(USE_CBLAS)
#define cblas_sgemm(...) assert(!"CBLAS is unavailable")
#endif

#else /* defined(USE_MKL) */

#define USE_MKL_PACKED_GEMM 0
#define USE_MKL_IGEMM 0

#if defined(_SX)
/* TODO: _SX should also define USE_CBLAS in case the later is available */
extern "C" {
#include "cblas.h" // CHECK: does SX also have a fortran API sgemm?
}

#elif defined(USE_CBLAS)
#include "cblas.h" // Maybe a system/cmake cblas works for you?
#else
/* put the stubs to make a code compilable but not workable */
#define cblas_sgemm(...) assert(!"CBLAS is unavailable")
#endif /* defined(_SX) */

#endif /* defined(USE_MKL) */

namespace mkldnn {
namespace impl {
namespace cpu {

#if defined(USE_MKL) && defined(USE_CBLAS)
typedef MKL_INT cblas_int;

#elif defined(USE_CBLAS)
typedef int cblas_int;

#if defined(_SX)
/* this cblas.h is peculiar... */
typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif
#endif

}
}
}

#endif /* OS_BLAS_HPP */

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
