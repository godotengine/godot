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

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <stdint.h>
#include <math.h>

#include "utils.hpp"
#include "nstl.hpp"
#include "mkldnn_traits.hpp"

#if defined(MKLDNN_X86_64)
#include "immintrin.h"
#endif

namespace mkldnn {
namespace impl {
namespace math {

/** rounds @p f to an integer according to the mxcsr register */
inline int mxcsr_round(float f) {
#if defined(MKLDNN_X86_64)
    return _mm_cvtss_si32(_mm_load_ss(&f));
#else
    return (int)nearbyintf(f); // optimism
#endif
}

template <typename data_t, typename acc_t>
inline typename utils::enable_if<!nstl::is_integral<data_t>::value,
       typename utils::remove_reference<data_t>::type>::type
saturate(const acc_t &x) {
    return (typename utils::remove_reference<data_t>::type)x;
}

template <typename data_t, typename acc_t>
inline typename utils::enable_if<nstl::is_integral<data_t>::value,
       typename utils::remove_reference<data_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    if (v < (acc_t)nstl::numeric_limits<data_t>::lowest())
        v = (acc_t)nstl::numeric_limits<data_t>::lowest();
    if (v > (acc_t)nstl::numeric_limits<data_t>::max())
        v = (acc_t)nstl::numeric_limits<data_t>::max();
    return (typename utils::remove_reference<data_t>::type)v;
}

template <typename data_t>
double saturate(const double &x) {
    double v = x;
    if (v < (double)nstl::numeric_limits<data_t>::lowest())
        v = (double)nstl::numeric_limits<data_t>::lowest();
    if (v > (double)nstl::numeric_limits<data_t>::max())
        v = (double)nstl::numeric_limits<data_t>::max();
    return v;
}

template <> inline int8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <> inline uint8_t saturate<uint8_t, int8_t>(const int8_t &x) {
    return x >= 0 ? x : 0;
}

template <typename out_t>
typename utils::enable_if<nstl::is_integral<out_t>::value, out_t>::type
out_round(float v) { return (out_t)mxcsr_round(v); }

template <typename out_t>
typename utils::enable_if<nstl::is_integral<out_t>::value, out_t>::type
out_round(double v) { return (out_t)mxcsr_round((float)v); }

template <typename out_t>
typename utils::enable_if<!nstl::is_integral<out_t>::value, out_t>::type
out_round(float v) { return v; }

inline int gcd(int a, int b) {
    a = impl::nstl::abs(a);
    b = impl::nstl::abs(b);
    if (a < b) { int x = a; a = b; b = x; }

    if (b == 0) return a;

    int r;
    while ((r = a % b) != 0) { a = b; b = r; }

    return b;
}

template <typename T>
inline bool is_pow2(const T& v) { return (v & (v - 1)) == 0; }

/** returns floor(log2(v)), aka the position of the leftmost non-0 bit */
inline int ilog2q(size_t v) {
    if (v == 0)
        return -1;

    int p = 0;
#   define CP(pw) do { if (v >= (1ull << pw)) { v >>= pw; p += pw; } } while(0)
    CP(32); CP(16); CP(8); CP(4); CP(2); CP(1);
#   undef CP
    return p;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U one_m_square(T x) {
    return (U)(1 - x) * (1 + x);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U x_m_square(T x) {
    return (U)(1 - x) * x;
}

/* activation */
template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U relu_fwd(T s, A alpha) {
    return s > 0 ? s : (U)(s * alpha);
}
template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U relu_bwd(T dd, T s, A alpha) {
    return s > 0 ? dd : (U)(dd * alpha);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U tanh_fwd(T s) {
    const float e = tanhf((float) s);
    return (U)e;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U tanh_bwd(T dd, T s) {
    const float e = tanh_fwd<float>((float) s);
    return (U)(dd * (1 - e) * (1 + e));
}

template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U elu_fwd(T s, A alpha) {
    return s > 0 ? s : (U)(alpha * (::expm1f((float)s)));
}
template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
 inline U elu_bwd(T dd, T s, A alpha) {
    return (U)(dd * (s > 0 ? 1 : alpha * ::expf((float)s)));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U square_fwd(T s) {
    return s * s;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U square_bwd(T dd, T s) {
    return dd * 2 * s;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U abs_fwd(T s) {
    return s > 0 ? s : -s;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U abs_bwd(T dd, T s) {
    return s > 0 ? dd : s < 0 ? -dd : 0;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U sqrt_fwd(T s) {
    return s > 0 ? (U)(::sqrtf((float)(s))) : 0;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U sqrt_bwd(T dd, T s) {
    return s > 0
        ? (U)(dd / (2 * ::sqrtf((float)(s))))
        : 0;
}

template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U linear_fwd(T s, A alpha, A beta) {
    return (U)(alpha * s + beta);
}

template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U linear_bwd(T dd, T s, A alpha, A beta) {
    (void) s;
    (void) beta;
    return (U)(dd * alpha);
}

template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U bounded_relu_fwd(T s, A alpha) {
    s = s > 0 ? s : 0;
    return s > alpha ? (U)(alpha) : s;
}

template <typename T, typename A,
         typename U = typename utils::remove_reference<T>::type>
inline U bounded_relu_bwd(T dd, T s, A alpha) {
    return dd * (0 < s && s < alpha ? 1 : 0);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U soft_relu_fwd(T s) {
    float max_logf = 8.872284e+01; //::logf(FLT_MAX)
    return s < max_logf ? (U)(::log1pf(::expf((float)s))) : s;
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U soft_relu_bwd(T dd, T s) {
    return (U)(dd / (1 + ::expf((float)(-s))));
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U logistic_fwd(T s) {
    U v = (U)(::expf((float) -s));
    return 1 / (1 + v);
}

template <typename T, typename U = typename utils::remove_reference<T>::type>
inline U logistic_bwd(T dd, T s) {
    U v = logistic_fwd<T, U>(s);
    return dd * v * (1 - v);
}

inline bool eltwise_fwd_preserves_zero(alg_kind_t alg, bool jit_impl = false) {
    using namespace alg_kind;
    using namespace utils;
    const bool preserves_zero = true
        && !one_of(alg, eltwise_linear, eltwise_soft_relu, eltwise_logistic)
        && IMPLICATION(jit_impl, !one_of(alg, eltwise_elu, eltwise_tanh));
    return preserves_zero;
}

inline float get_bias(const char *bias, size_t offset, data_type_t data_type)
{
    if (!bias)
        return 0.0f;

#define CASE(dt) \
    case dt: return (float)((const prec_traits<dt>::type *)bias)[offset]

    switch (data_type) {
    CASE(data_type::s8);
    CASE(data_type::u8);
    CASE(data_type::s32);
    CASE(data_type::f32);
    default: assert(!"unimplemented");
    }
    return 0; // never happens (should probably be a NaN)
#undef CASE
}

}
}
}

#endif
