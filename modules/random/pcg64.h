/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
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
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

#ifndef PCG64_H_INCLUDED
#define PCG64_H_INCLUDED 1

#include <inttypes.h>

#if __GNUC_GNU_INLINE__  &&  !defined(__cplusplus)
    #error Nonstandard GNU inlining semantics. Compile with -std=c99 or better.
#endif

#if __cplusplus
extern "C" {
#endif

#if __SIZEOF_INT128__ && !defined(PCG_FORCE_EMULATED_128BIT_MATH)
    typedef __uint128_t pcg128_t;
    #define PCG_128BIT_CONSTANT(high,low) \
            (((pcg128_t)(high) << 64) + low)
#else
    typedef struct {
        uint64_t high;
        uint64_t low;
    } pcg128_t;

    inline pcg128_t PCG_128BIT_CONSTANT(uint64_t high, uint64_t low) {
        pcg128_t result;
        result.high = high;
        result.low = low;
        return result;
    }

    #define PCG_EMULATED_128BIT_MATH 1
#endif

typedef struct {
    pcg128_t state;
} pcg_state_128;

typedef struct {
    pcg128_t state;
    pcg128_t inc;
} pcg_state_setseq_128;

#define PCG_DEFAULT_MULTIPLIER_128 \
        PCG_128BIT_CONSTANT(2549297995355413924ULL,4865540595714422341ULL)
#define PCG_DEFAULT_INCREMENT_128  \
        PCG_128BIT_CONSTANT(6364136223846793005ULL,1442695040888963407ULL)
#define PCG_STATE_SETSEQ_128_INITIALIZER                                       \
    { PCG_128BIT_CONSTANT(0x979c9a98d8462005ULL, 0x7d3e9cb6cfe0549bULL),       \
      PCG_128BIT_CONSTANT(0x0000000000000001ULL, 0xda3e39cb94b95bdbULL) }

inline uint64_t pcg_rotr_64(uint64_t value, unsigned int rot)
{
    return (value >> rot) | (value << ((- rot) & 63));
}

#ifdef PCG_EMULATED_128BIT_MATH

inline pcg128_t _pcg128_add(pcg128_t a, pcg128_t b) {
    pcg128_t result;

    result.low = a.low + b.low;
    result.high = a.high + b.high + (result.low < b.low);
    return result;
}

inline void _pcg_mult64(uint64_t x, uint64_t y, uint64_t* z1, uint64_t* z0) {
    uint64_t x0, x1, y0, y1;
    uint64_t w0, w1, w2, t;
    /* Lower 64 bits are straightforward clock-arithmetic. */
    *z0 = x * y;

    x0 = x & 0xFFFFFFFFULL;
    x1 = x >> 32;
    y0 = y & 0xFFFFFFFFULL;
    y1 = y >> 32;
    w0 = x0 * y0;
    t = x1 * y0 + (w0 >> 32);
    w1 = t & 0xFFFFFFFFULL;
    w2 = t >> 32;
    w1 += x0 * y1;
    *z1 = x1 * y1 + w2 + (w1 >> 32);
}

inline pcg128_t _pcg128_mult(pcg128_t a, pcg128_t b) {
    uint64_t h1;
    pcg128_t result;

    h1 = a.high * b.low + a.low * b.high;
    _pcg_mult64(a.low, b.low, &(result.high), &(result.low));
    result.high += h1;
    return result;
}

inline void pcg_setseq_128_step_r(pcg_state_setseq_128* rng)
{
    rng->state = _pcg128_add(_pcg128_mult(rng->state, PCG_DEFAULT_MULTIPLIER_128), rng->inc);
}

inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state)
{
    return pcg_rotr_64(state.high ^ state.low,
                       state.high >> 58u);
}

inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128* rng,
                                     pcg128_t initstate, pcg128_t initseq)
{
    rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
    rng->inc.high = initseq.high << 1u;
    rng->inc.high |= initseq.low & 0x800000000000ULL;
    rng->inc.low = (initseq.low << 1u) | 1u;
    pcg_setseq_128_step_r(rng);
    rng->state = _pcg128_add(rng->state, initstate);
    pcg_setseq_128_step_r(rng);
}

#else /* PCG_EMULATED_128BIT_MATH */

inline void pcg_setseq_128_step_r(pcg_state_setseq_128* rng)
{
    rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_128 + rng->inc;
}

inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state)
{
    return pcg_rotr_64(((uint64_t)(state >> 64u)) ^ (uint64_t)state,
                       state >> 122u);
}

inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128* rng,
                                     pcg128_t initstate, pcg128_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg_setseq_128_step_r(rng);
    rng->state += initstate;
    pcg_setseq_128_step_r(rng);
}

#endif /* PCG_EMULATED_128BIT_MATH */


inline uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128* rng)
{
    pcg_setseq_128_step_r(rng);
    return pcg_output_xsl_rr_128_64(rng->state);
}

inline uint64_t
pcg_setseq_128_xsl_rr_64_boundedrand_r(pcg_state_setseq_128* rng,
                                       uint64_t bound)
{
    uint64_t threshold = -bound % bound;
    for (;;) {
        uint64_t r = pcg_setseq_128_xsl_rr_64_random_r(rng);
        if (r >= threshold)
            return r % bound;
    }
}

extern pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                                    pcg128_t cur_plus);

inline void pcg_setseq_128_advance_r(pcg_state_setseq_128* rng, pcg128_t delta)
{
    rng->state = pcg_advance_lcg_128(rng->state, delta,
                                     PCG_DEFAULT_MULTIPLIER_128, rng->inc);
}

typedef pcg_state_setseq_128    pcg64_random_t;
#define pcg64_random_r          pcg_setseq_128_xsl_rr_64_random_r
#define pcg64_boundedrand_r     pcg_setseq_128_xsl_rr_64_boundedrand_r
#define pcg64_srandom_r         pcg_setseq_128_srandom_r
#define pcg64_advance_r         pcg_setseq_128_advance_r
#define PCG64_INITIALIZER       PCG_STATE_SETSEQ_128_INITIALIZER

#if __cplusplus
}
#endif

#endif /* PCG64_H_INCLUDED */
