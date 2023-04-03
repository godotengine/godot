/*
 * License for Berkeley SoftFloat Release 3e
 *
 * John R. Hauser
 * 2018 January 20
 *
 * The following applies to the whole of SoftFloat Release 3e as well as to
 * each source file individually.
 *
 * Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018 The Regents of the
 * University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions, and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions, and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the University nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * The functions listed in this file are modified versions of the ones
 * from the Berkeley SoftFloat 3e Library.
 *
 * Their implementation correctness has been checked with the Berkeley
 * TestFloat Release 3e tool for x86_64.
 */

#include "rounding.h"
#include "bitscan.h"
#include "softfloat.h"

#if defined(BIG_ENDIAN)
#define word_incr -1
#define index_word(total, n) ((total) - 1 - (n))
#define index_word_hi(total) 0
#define index_word_lo(total) ((total) - 1)
#define index_multiword_hi(total, n) 0
#define index_multiword_lo(total, n) ((total) - (n))
#define index_multiword_hi_but(total, n) 0
#define index_multiword_lo_but(total, n) (n)
#else
#define word_incr 1
#define index_word(total, n) (n)
#define index_word_hi(total) ((total) - 1)
#define index_word_lo(total) 0
#define index_multiword_hi(total, n) ((total) - (n))
#define index_multiword_lo(total, n) 0
#define index_multiword_hi_but(total, n) (n)
#define index_multiword_lo_but(total, n) 0
#endif

typedef union { double f; int64_t i; uint64_t u; } di_type;
typedef union { float f; int32_t i; uint32_t u; } fi_type;

const uint8_t count_leading_zeros8[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/**
 * \brief Shifts 'a' right by the number of bits given in 'dist', which must be in
 * the range 1 to 63.  If any nonzero bits are shifted off, they are "jammed"
 * into the least-significant bit of the shifted value by setting the
 * least-significant bit to 1.  This shifted-and-jammed value is returned.
 *
 * From softfloat_shortShiftRightJam64()
 */
static inline
uint64_t _mesa_short_shift_right_jam64(uint64_t a, uint8_t dist)
{
    return a >> dist | ((a & (((uint64_t) 1 << dist) - 1)) != 0);
}

/**
 * \brief Shifts 'a' right by the number of bits given in 'dist', which must not
 * be zero.  If any nonzero bits are shifted off, they are "jammed" into the
 * least-significant bit of the shifted value by setting the least-significant
 * bit to 1.  This shifted-and-jammed value is returned.
 * The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
 * greater than 64, the result will be either 0 or 1, depending on whether 'a'
 * is zero or nonzero.
 *
 * From softfloat_shiftRightJam64()
 */
static inline
uint64_t _mesa_shift_right_jam64(uint64_t a, uint32_t dist)
{
    return
        (dist < 63) ? a >> dist | ((uint64_t) (a << (-dist & 63)) != 0) : (a != 0);
}

/**
 * \brief Shifts 'a' right by the number of bits given in 'dist', which must not be
 * zero.  If any nonzero bits are shifted off, they are "jammed" into the
 * least-significant bit of the shifted value by setting the least-significant
 * bit to 1.  This shifted-and-jammed value is returned.
 * The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
 * greater than 32, the result will be either 0 or 1, depending on whether 'a'
 * is zero or nonzero.
 *
 * From softfloat_shiftRightJam32()
 */
static inline
uint32_t _mesa_shift_right_jam32(uint32_t a, uint16_t dist)
{
    return
        (dist < 31) ? a >> dist | ((uint32_t) (a << (-dist & 31)) != 0) : (a != 0);
}

/**
 * \brief Extracted from softfloat_roundPackToF64()
 */
static inline
double _mesa_roundtozero_f64(int64_t s, int64_t e, int64_t m)
{
    di_type result;

    if ((uint64_t) e >= 0x7fd) {
        if (e < 0) {
            m = _mesa_shift_right_jam64(m, -e);
            e = 0;
        } else if ((e > 0x7fd) || (0x8000000000000000 <= m)) {
            e = 0x7ff;
            m = 0;
            result.u = (s << 63) + (e << 52) + m;
            result.u -= 1;
            return result.f;
        }
    }

    m >>= 10;
    if (m == 0)
        e = 0;

    result.u = (s << 63) + (e << 52) + m;
    return result.f;
}

/**
 * \brief Extracted from softfloat_roundPackToF32()
 */
static inline
float _mesa_round_f32(int32_t s, int32_t e, int32_t m, bool rtz)
{
    fi_type result;
    uint8_t round_increment = rtz ? 0 : 0x40;

    if ((uint32_t) e >= 0xfd) {
        if (e < 0) {
            m = _mesa_shift_right_jam32(m, -e);
            e = 0;
        } else if ((e > 0xfd) || (0x80000000 <= m + round_increment)) {
            e = 0xff;
            m = 0;
            result.u = (s << 31) + (e << 23) + m;
            result.u -= !round_increment;
            return result.f;
        }
    }

    uint8_t round_bits;
    round_bits = m & 0x7f;
    m = ((uint32_t) m + round_increment) >> 7;
    m &= ~(uint32_t) (! (round_bits ^ 0x40) & !rtz);
    if (m == 0)
        e = 0;

    result.u = (s << 31) + (e << 23) + m;
    return result.f;
}

/**
 * \brief Extracted from softfloat_roundPackToF16()
 */
static inline
uint16_t _mesa_roundtozero_f16(int16_t s, int16_t e, int16_t m)
{
    if ((uint16_t) e >= 0x1d) {
        if (e < 0) {
            m = _mesa_shift_right_jam32(m, -e);
            e = 0;
        } else if (e > 0x1d) {
            e = 0x1f;
            m = 0;
            return (s << 15) + (e << 10) + m - 1;
        }
    }

    m >>= 4;
    if (m == 0)
        e = 0;

    return (s << 15) + (e << 10) + m;
}

/**
 * \brief Shifts the N-bit unsigned integer pointed to by 'a' left by the number of
 * bits given in 'dist', where N = 'size_words' * 32.  The value of 'dist'
 * must be in the range 1 to 31.  Any nonzero bits shifted off are lost.  The
 * shifted N-bit result is stored at the location pointed to by 'm_out'.  Each
 * of 'a' and 'm_out' points to a 'size_words'-long array of 32-bit elements
 * that concatenate in the platform's normal endian order to form an N-bit
 * integer.
 *
 * From softfloat_shortShiftLeftM()
 */
static inline void
_mesa_short_shift_left_m(uint8_t size_words, const uint32_t *a, uint8_t dist, uint32_t *m_out)
{
    uint8_t neg_dist;
    unsigned index, last_index;
    uint32_t part_word, a_word;

    neg_dist = -dist;
    index = index_word_hi(size_words);
    last_index = index_word_lo(size_words);
    part_word = a[index] << dist;
    while (index != last_index) {
        a_word = a[index - word_incr];
        m_out[index] = part_word | a_word >> (neg_dist & 31);
        index -= word_incr;
        part_word = a_word << dist;
    }
    m_out[index] = part_word;
}

/**
 * \brief Shifts the N-bit unsigned integer pointed to by 'a' left by the number of
 * bits given in 'dist', where N = 'size_words' * 32.  The value of 'dist'
 * must not be zero.  Any nonzero bits shifted off are lost.  The shifted
 * N-bit result is stored at the location pointed to by 'm_out'.  Each of 'a'
 * and 'm_out' points to a 'size_words'-long array of 32-bit elements that
 * concatenate in the platform's normal endian order to form an N-bit
 * integer. The value of 'dist' can be arbitrarily large.  In particular, if
 * 'dist' is greater than N, the stored result will be 0.
 *
 * From softfloat_shiftLeftM()
 */
static inline void
_mesa_shift_left_m(uint8_t size_words, const uint32_t *a, uint32_t dist, uint32_t *m_out)
{
    uint32_t word_dist;
    uint8_t inner_dist;
    uint8_t i;

    word_dist = dist >> 5;
    if (word_dist < size_words) {
        a += index_multiword_lo_but(size_words, word_dist);
        inner_dist = dist & 31;
        if (inner_dist) {
            _mesa_short_shift_left_m(size_words - word_dist, a, inner_dist,
                                     m_out + index_multiword_hi_but(size_words, word_dist));
            if (!word_dist)
                return;
        } else {
            uint32_t *dest = m_out + index_word_hi(size_words);
            a += index_word_hi(size_words - word_dist);
            for (i = size_words - word_dist; i; --i) {
                *dest = *a;
                a -= word_incr;
                dest -= word_incr;
            }
        }
        m_out += index_multiword_lo(size_words, word_dist);
    } else {
        word_dist = size_words;
    }
    do {
        *m_out++ = 0;
        --word_dist;
    } while (word_dist);
}

/**
 * \brief Shifts the N-bit unsigned integer pointed to by 'a' right by the number of
 * bits given in 'dist', where N = 'size_words' * 32.  The value of 'dist'
 * must be in the range 1 to 31.  Any nonzero bits shifted off are lost.  The
 * shifted N-bit result is stored at the location pointed to by 'm_out'.  Each
 * of 'a' and 'm_out' points to a 'size_words'-long array of 32-bit elements
 * that concatenate in the platform's normal endian order to form an N-bit
 * integer.
 *
 * From softfloat_shortShiftRightM()
 */
static inline void
_mesa_short_shift_right_m(uint8_t size_words, const uint32_t *a, uint8_t dist, uint32_t *m_out)
{
    uint8_t neg_dist;
    unsigned index, last_index;
    uint32_t part_word, a_word;

    neg_dist = -dist;
    index = index_word_lo(size_words);
    last_index = index_word_hi(size_words);
    part_word = a[index] >> dist;
    while (index != last_index) {
        a_word = a[index + word_incr];
        m_out[index] = a_word << (neg_dist & 31) | part_word;
        index += word_incr;
        part_word = a_word >> dist;
    }
    m_out[index] = part_word;
}

/**
 * \brief Shifts the N-bit unsigned integer pointed to by 'a' right by the number of
 * bits given in 'dist', where N = 'size_words' * 32.  The value of 'dist'
 * must be in the range 1 to 31.  If any nonzero bits are shifted off, they
 * are "jammed" into the least-significant bit of the shifted value by setting
 * the least-significant bit to 1.  This shifted-and-jammed N-bit result is
 * stored at the location pointed to by 'm_out'.  Each of 'a' and 'm_out'
 * points to a 'size_words'-long array of 32-bit elements that concatenate in
 * the platform's normal endian order to form an N-bit integer.
 *
 *
 * From softfloat_shortShiftRightJamM()
 */
static inline void
_mesa_short_shift_right_jam_m(uint8_t size_words, const uint32_t *a, uint8_t dist, uint32_t *m_out)
{
    uint8_t neg_dist;
    unsigned index, last_index;
    uint64_t part_word, a_word;

    neg_dist = -dist;
    index = index_word_lo(size_words);
    last_index = index_word_hi(size_words);
    a_word = a[index];
    part_word = a_word >> dist;
    if (part_word << dist != a_word )
        part_word |= 1;
    while (index != last_index) {
        a_word = a[index + word_incr];
        m_out[index] = a_word << (neg_dist & 31) | part_word;
        index += word_incr;
        part_word = a_word >> dist;
    }
    m_out[index] = part_word;
}

/**
 * \brief Shifts the N-bit unsigned integer pointed to by 'a' right by the number of
 * bits given in 'dist', where N = 'size_words' * 32.  The value of 'dist'
 * must not be zero.  If any nonzero bits are shifted off, they are "jammed"
 * into the least-significant bit of the shifted value by setting the
 * least-significant bit to 1.  This shifted-and-jammed N-bit result is stored
 * at the location pointed to by 'm_out'.  Each of 'a' and 'm_out' points to a
 * 'size_words'-long array of 32-bit elements that concatenate in the
 * platform's normal endian order to form an N-bit integer.  The value of
 * 'dist' can be arbitrarily large.  In particular, if 'dist' is greater than
 * N, the stored result will be either 0 or 1, depending on whether the
 * original N bits are all zeros.
 *
 * From softfloat_shiftRightJamM()
 */
static inline void
_mesa_shift_right_jam_m(uint8_t size_words, const uint32_t *a, uint32_t dist, uint32_t *m_out)
{
    uint32_t word_jam, word_dist, *tmp;
    uint8_t i, inner_dist;

    word_jam = 0;
    word_dist = dist >> 5;
    tmp = NULL;
    if (word_dist) {
        if (size_words < word_dist)
            word_dist = size_words;
        tmp = (uint32_t *) (a + index_multiword_lo(size_words, word_dist));
        i = word_dist;
        do {
            word_jam = *tmp++;
            if (word_jam)
                break;
            --i;
        } while (i);
        tmp = m_out;
    }
    if (word_dist < size_words) {
        a += index_multiword_hi_but(size_words, word_dist);
        inner_dist = dist & 31;
        if (inner_dist) {
            _mesa_short_shift_right_jam_m(size_words - word_dist, a, inner_dist,
                                          m_out + index_multiword_lo_but(size_words, word_dist));
            if (!word_dist) {
                if (word_jam)
                    m_out[index_word_lo(size_words)] |= 1;
                return;
            }
        } else {
            a += index_word_lo(size_words - word_dist);
            tmp = m_out + index_word_lo(size_words);
            for (i = size_words - word_dist; i; --i) {
                *tmp = *a;
                a += word_incr;
                tmp += word_incr;
            }
        }
        tmp = m_out + index_multiword_hi(size_words, word_dist);
    }
    if (tmp) {
       do {
           *tmp++ = 0;
           --word_dist;
       } while (word_dist);
    }
    if (word_jam)
        m_out[index_word_lo(size_words)] |= 1;
}

/**
 * \brief Calculate a + b but rounding to zero.
 *
 * Notice that this mainly differs from the original Berkeley SoftFloat 3e
 * implementation in that we don't really treat NaNs, Zeroes nor the
 * signalling flags. Any NaN is good for us and the sign of the Zero is not
 * important.
 *
 * From f64_add()
 */
double
_mesa_double_add_rtz(double a, double b)
{
    const di_type a_di = {a};
    uint64_t a_flt_m = a_di.u & 0x0fffffffffffff;
    uint64_t a_flt_e = (a_di.u >> 52) & 0x7ff;
    uint64_t a_flt_s = (a_di.u >> 63) & 0x1;
    const di_type b_di = {b};
    uint64_t b_flt_m = b_di.u & 0x0fffffffffffff;
    uint64_t b_flt_e = (b_di.u >> 52) & 0x7ff;
    uint64_t b_flt_s = (b_di.u >> 63) & 0x1;
    int64_t s, e, m = 0;

    s = a_flt_s;

    const int64_t exp_diff = a_flt_e - b_flt_e;

    /* Handle special cases */

    if (a_flt_s != b_flt_s) {
        return _mesa_double_sub_rtz(a, -b);
    } else if ((a_flt_e == 0) && (a_flt_m == 0)) {
        /* 'a' is zero, return 'b' */
        return b;
    } else if ((b_flt_e == 0) && (b_flt_m == 0)) {
        /* 'b' is zero, return 'a' */
        return a;
    } else if (a_flt_e == 0x7ff && a_flt_m != 0) {
        /* 'a' is a NaN, return NaN */
        return a;
    } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
        /* 'b' is a NaN, return NaN */
        return b;
    } else if (a_flt_e == 0x7ff && a_flt_m == 0) {
        /* Inf + x = Inf */
        return a;
    } else if (b_flt_e == 0x7ff && b_flt_m == 0) {
        /* x + Inf = Inf */
        return b;
    } else if (exp_diff == 0 && a_flt_e == 0) {
        di_type result_di;
        result_di.u = a_di.u + b_flt_m;
        return result_di.f;
    } else if (exp_diff == 0) {
        e = a_flt_e;
        m = 0x0020000000000000 + a_flt_m + b_flt_m;
        m <<= 9;
    } else if (exp_diff < 0) {
        a_flt_m <<= 9;
        b_flt_m <<= 9;
        e = b_flt_e;

        if (a_flt_e != 0)
            a_flt_m += 0x2000000000000000;
        else
            a_flt_m <<= 1;

        a_flt_m = _mesa_shift_right_jam64(a_flt_m, -exp_diff);
        m = 0x2000000000000000 + a_flt_m + b_flt_m;
        if (m < 0x4000000000000000) {
            --e;
            m <<= 1;
        }
    } else {
        a_flt_m <<= 9;
        b_flt_m <<= 9;
        e = a_flt_e;

        if (b_flt_e != 0)
            b_flt_m += 0x2000000000000000;
        else
            b_flt_m <<= 1;

        b_flt_m = _mesa_shift_right_jam64(b_flt_m, exp_diff);
        m = 0x2000000000000000 + a_flt_m + b_flt_m;
        if (m < 0x4000000000000000) {
            --e;
            m <<= 1;
        }
    }

    return _mesa_roundtozero_f64(s, e, m);
}

/**
 * \brief Returns the number of leading 0 bits before the most-significant 1 bit of
 * 'a'.  If 'a' is zero, 64 is returned.
 */
static inline unsigned
_mesa_count_leading_zeros64(uint64_t a)
{
    return 64 - util_last_bit64(a);
}

/**
 * \brief Returns the number of leading 0 bits before the most-significant 1 bit of
 * 'a'.  If 'a' is zero, 32 is returned.
 */
static inline unsigned
_mesa_count_leading_zeros32(uint32_t a)
{
    return 32 - util_last_bit(a);
}

static inline double
_mesa_norm_round_pack_f64(int64_t s, int64_t e, int64_t m)
{
    int8_t shift_dist;

    shift_dist = _mesa_count_leading_zeros64(m) - 1;
    e -= shift_dist;
    if ((10 <= shift_dist) && ((unsigned) e < 0x7fd)) {
        di_type result;
        result.u = (s << 63) + ((m ? e : 0) << 52) + (m << (shift_dist - 10));
        return result.f;
    } else {
        return _mesa_roundtozero_f64(s, e, m << shift_dist);
    }
}

/**
 * \brief Replaces the N-bit unsigned integer pointed to by 'm_out' by the
 * 2s-complement of itself, where N = 'size_words' * 32.  Argument 'm_out'
 * points to a 'size_words'-long array of 32-bit elements that concatenate in
 * the platform's normal endian order to form an N-bit integer.
 *
 * From softfloat_negXM()
 */
static inline void
_mesa_neg_x_m(uint8_t size_words, uint32_t *m_out)
{
    unsigned index, last_index;
    uint8_t carry;
    uint32_t word;

    index = index_word_lo(size_words);
    last_index = index_word_hi(size_words);
    carry = 1;
    for (;;) {
        word = ~m_out[index] + carry;
        m_out[index] = word;
        if (index == last_index)
            break;
        index += word_incr;
        if (word)
            carry = 0;
    }
}

/**
 * \brief Adds the two N-bit integers pointed to by 'a' and 'b', where N =
 * 'size_words' * 32.  The addition is modulo 2^N, so any carry out is
 * lost. The N-bit sum is stored at the location pointed to by 'm_out'.  Each
 * of 'a', 'b', and 'm_out' points to a 'size_words'-long array of 32-bit
 * elements that concatenate in the platform's normal endian order to form an
 * N-bit integer.
 *
 * From softfloat_addM()
 */
static inline void
_mesa_add_m(uint8_t size_words, const uint32_t *a, const uint32_t *b, uint32_t *m_out)
{
    unsigned index, last_index;
    uint8_t carry;
    uint32_t a_word, word;

    index = index_word_lo(size_words);
    last_index = index_word_hi(size_words);
    carry = 0;
    for (;;) {
        a_word = a[index];
        word = a_word + b[index] + carry;
        m_out[index] = word;
        if (index == last_index)
            break;
        if (word != a_word)
            carry = (word < a_word);
        index += word_incr;
    }
}

/**
 * \brief Subtracts the two N-bit integers pointed to by 'a' and 'b', where N =
 * 'size_words' * 32.  The subtraction is modulo 2^N, so any borrow out (carry
 * out) is lost.  The N-bit difference is stored at the location pointed to by
 * 'm_out'.  Each of 'a', 'b', and 'm_out' points to a 'size_words'-long array
 * of 32-bit elements that concatenate in the platform's normal endian order
 * to form an N-bit integer.
 *
 * From softfloat_subM()
 */
static inline void
_mesa_sub_m(uint8_t size_words, const uint32_t *a, const uint32_t *b, uint32_t *m_out)
{
    unsigned index, last_index;
    uint8_t borrow;
    uint32_t a_word, b_word;

    index = index_word_lo(size_words);
    last_index = index_word_hi(size_words);
    borrow = 0;
    for (;;) {
        a_word = a[index];
        b_word = b[index];
        m_out[index] = a_word - b_word - borrow;
        if (index == last_index)
            break;
        borrow = borrow ? (a_word <= b_word) : (a_word < b_word);
        index += word_incr;
    }
}

/* Calculate a - b but rounding to zero.
 *
 * Notice that this mainly differs from the original Berkeley SoftFloat 3e
 * implementation in that we don't really treat NaNs, Zeroes nor the
 * signalling flags. Any NaN is good for us and the sign of the Zero is not
 * important.
 *
 * From f64_sub()
 */
double
_mesa_double_sub_rtz(double a, double b)
{
    const di_type a_di = {a};
    uint64_t a_flt_m = a_di.u & 0x0fffffffffffff;
    uint64_t a_flt_e = (a_di.u >> 52) & 0x7ff;
    uint64_t a_flt_s = (a_di.u >> 63) & 0x1;
    const di_type b_di = {b};
    uint64_t b_flt_m = b_di.u & 0x0fffffffffffff;
    uint64_t b_flt_e = (b_di.u >> 52) & 0x7ff;
    uint64_t b_flt_s = (b_di.u >> 63) & 0x1;
    int64_t s, e, m = 0;
    int64_t m_diff = 0;
    unsigned shift_dist = 0;

    s = a_flt_s;

    const int64_t exp_diff = a_flt_e - b_flt_e;

    /* Handle special cases */

    if (a_flt_s != b_flt_s) {
        return _mesa_double_add_rtz(a, -b);
    } else if ((a_flt_e == 0) && (a_flt_m == 0)) {
        /* 'a' is zero, return '-b' */
        return -b;
    } else if ((b_flt_e == 0) && (b_flt_m == 0)) {
        /* 'b' is zero, return 'a' */
        return a;
    } else if (a_flt_e == 0x7ff && a_flt_m != 0) {
        /* 'a' is a NaN, return NaN */
        return a;
    } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
        /* 'b' is a NaN, return NaN */
        return b;
    } else if (a_flt_e == 0x7ff && a_flt_m == 0) {
        if (b_flt_e == 0x7ff && b_flt_m == 0) {
            /* Inf - Inf =  NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }
        /* Inf - x = Inf */
        return a;
    } else if (b_flt_e == 0x7ff && b_flt_m == 0) {
        /* x - Inf = -Inf */
        return -b;
    } else if (exp_diff == 0) {
        m_diff = a_flt_m - b_flt_m;

        if (m_diff == 0)
            return 0;
        if (a_flt_e)
            --a_flt_e;
        if (m_diff < 0) {
            s = !s;
            m_diff = -m_diff;
        }

        shift_dist = _mesa_count_leading_zeros64(m_diff) - 11;
        e = a_flt_e - shift_dist;
        if (e < 0) {
            shift_dist = a_flt_e;
            e = 0;
        }

        di_type result;
        result.u = (s << 63) + (e << 52) + (m_diff << shift_dist);
        return result.f;
    } else if (exp_diff < 0) {
        a_flt_m <<= 10;
        b_flt_m <<= 10;
        s = !s;

        a_flt_m += (a_flt_e) ? 0x4000000000000000 : a_flt_m;
        a_flt_m = _mesa_shift_right_jam64(a_flt_m, -exp_diff);
        b_flt_m |= 0x4000000000000000;
        e = b_flt_e;
        m = b_flt_m - a_flt_m;
    } else {
        a_flt_m <<= 10;
        b_flt_m <<= 10;

        b_flt_m += (b_flt_e) ? 0x4000000000000000 : b_flt_m;
        b_flt_m = _mesa_shift_right_jam64(b_flt_m, exp_diff);
        a_flt_m |= 0x4000000000000000;
        e = a_flt_e;
        m = a_flt_m - b_flt_m;
    }

    return _mesa_norm_round_pack_f64(s, e - 1, m);
}

static inline void
_mesa_norm_subnormal_mantissa_f64(uint64_t m, uint64_t *exp, uint64_t *m_out)
{
    int shift_dist;

    shift_dist = _mesa_count_leading_zeros64(m) - 11;
    *exp = 1 - shift_dist;
    *m_out = m << shift_dist;
}

static inline void
_mesa_norm_subnormal_mantissa_f32(uint32_t m, uint32_t *exp, uint32_t *m_out)
{
    int shift_dist;

    shift_dist = _mesa_count_leading_zeros32(m) - 8;
    *exp = 1 - shift_dist;
    *m_out = m << shift_dist;
}

/**
 * \brief Multiplies 'a' and 'b' and stores the 128-bit product at the location
 * pointed to by 'zPtr'.  Argument 'zPtr' points to an array of four 32-bit
 * elements that concatenate in the platform's normal endian order to form a
 * 128-bit integer.
 *
 * From softfloat_mul64To128M()
 */
static inline void
_mesa_softfloat_mul_f64_to_f128_m(uint64_t a, uint64_t b, uint32_t *m_out)
{
    uint32_t a32, a0, b32, b0;
    uint64_t z0, mid1, z64, mid;

    a32 = a >> 32;
    a0 = a;
    b32 = b >> 32;
    b0 = b;
    z0 = (uint64_t) a0 * b0;
    mid1 = (uint64_t) a32 * b0;
    mid = mid1 + (uint64_t) a0 * b32;
    z64 = (uint64_t) a32 * b32;
    z64 += (uint64_t) (mid < mid1) << 32 | mid >> 32;
    mid <<= 32;
    z0 += mid;
    m_out[index_word(4, 1)] = z0 >> 32;
    m_out[index_word(4, 0)] = z0;
    z64 += (z0 < mid);
    m_out[index_word(4, 3)] = z64 >> 32;
    m_out[index_word(4, 2)] = z64;
}

/* Calculate a * b but rounding to zero.
 *
 * Notice that this mainly differs from the original Berkeley SoftFloat 3e
 * implementation in that we don't really treat NaNs, Zeroes nor the
 * signalling flags. Any NaN is good for us and the sign of the Zero is not
 * important.
 *
 * From f64_mul()
 */
double
_mesa_double_mul_rtz(double a, double b)
{
    const di_type a_di = {a};
    uint64_t a_flt_m = a_di.u & 0x0fffffffffffff;
    uint64_t a_flt_e = (a_di.u >> 52) & 0x7ff;
    uint64_t a_flt_s = (a_di.u >> 63) & 0x1;
    const di_type b_di = {b};
    uint64_t b_flt_m = b_di.u & 0x0fffffffffffff;
    uint64_t b_flt_e = (b_di.u >> 52) & 0x7ff;
    uint64_t b_flt_s = (b_di.u >> 63) & 0x1;
    int64_t s, e, m = 0;

    s = a_flt_s ^ b_flt_s;

    if (a_flt_e == 0x7ff) {
        if (a_flt_m != 0) {
            /* 'a' is a NaN, return NaN */
            return a;
        } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        }

        if (!(b_flt_e | b_flt_m)) {
            /* Inf * 0 = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }
        /* Inf * x = Inf */
        di_type result;
        e = 0x7ff;
        result.u = (s << 63) + (e << 52) + 0;
        return result.f;
    }

    if (b_flt_e == 0x7ff) {
        if (b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        }
        if (!(a_flt_e | a_flt_m)) {
            /* 0 * Inf = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }
        /* x * Inf = Inf */
        di_type result;
        e = 0x7ff;
        result.u = (s << 63) + (e << 52) + 0;
        return result.f;
    }

    if (a_flt_e == 0) {
        if (a_flt_m == 0) {
            /* 'a' is zero. Return zero */
            di_type result;
            result.u = (s << 63) + 0;
            return result.f;
        }
        _mesa_norm_subnormal_mantissa_f64(a_flt_m , &a_flt_e, &a_flt_m);
    }
    if (b_flt_e == 0) {
        if (b_flt_m == 0) {
            /* 'b' is zero. Return zero */
            di_type result;
            result.u = (s << 63) + 0;
            return result.f;
        }
        _mesa_norm_subnormal_mantissa_f64(b_flt_m , &b_flt_e, &b_flt_m);
    }

    e = a_flt_e + b_flt_e - 0x3ff;
    a_flt_m = (a_flt_m | 0x0010000000000000) << 10;
    b_flt_m = (b_flt_m | 0x0010000000000000) << 11;

    uint32_t m_128[4];
    _mesa_softfloat_mul_f64_to_f128_m(a_flt_m, b_flt_m, m_128);

    m = (uint64_t) m_128[index_word(4, 3)] << 32 | m_128[index_word(4, 2)];
    if (m_128[index_word(4, 1)] || m_128[index_word(4, 0)])
        m |= 1;

    if (m < 0x4000000000000000) {
        --e;
        m <<= 1;
    }

    return _mesa_roundtozero_f64(s, e, m);
}


/**
 * \brief Calculate a * b + c but rounding to zero.
 *
 * Notice that this mainly differs from the original Berkeley SoftFloat 3e
 * implementation in that we don't really treat NaNs, Zeroes nor the
 * signalling flags. Any NaN is good for us and the sign of the Zero is not
 * important.
 *
 * From f64_mulAdd()
 */
double
_mesa_double_fma_rtz(double a, double b, double c)
{
    const di_type a_di = {a};
    uint64_t a_flt_m = a_di.u & 0x0fffffffffffff;
    uint64_t a_flt_e = (a_di.u >> 52) & 0x7ff;
    uint64_t a_flt_s = (a_di.u >> 63) & 0x1;
    const di_type b_di = {b};
    uint64_t b_flt_m = b_di.u & 0x0fffffffffffff;
    uint64_t b_flt_e = (b_di.u >> 52) & 0x7ff;
    uint64_t b_flt_s = (b_di.u >> 63) & 0x1;
    const di_type c_di = {c};
    uint64_t c_flt_m = c_di.u & 0x0fffffffffffff;
    uint64_t c_flt_e = (c_di.u >> 52) & 0x7ff;
    uint64_t c_flt_s = (c_di.u >> 63) & 0x1;
    int64_t s, e, m = 0;

    c_flt_s ^= 0;
    s = a_flt_s ^ b_flt_s ^ 0;

    if (a_flt_e == 0x7ff) {
        if (a_flt_m != 0) {
            /* 'a' is a NaN, return NaN */
            return a;
        } else if (b_flt_e == 0x7ff && b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        } else if (c_flt_e == 0x7ff && c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        if (!(b_flt_e | b_flt_m)) {
            /* Inf * 0 + y = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }

        if ((c_flt_e == 0x7ff && c_flt_m == 0) && (s != c_flt_s)) {
            /* Inf * x - Inf = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }

        /* Inf * x + y = Inf */
        di_type result;
        e = 0x7ff;
        result.u = (s << 63) + (e << 52) + 0;
        return result.f;
    }

    if (b_flt_e == 0x7ff) {
        if (b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        } else if (c_flt_e == 0x7ff && c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        if (!(a_flt_e | a_flt_m)) {
            /* 0 * Inf + y = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }

        if ((c_flt_e == 0x7ff && c_flt_m == 0) && (s != c_flt_s)) {
            /* x * Inf - Inf = NaN */
            di_type result;
            e = 0x7ff;
            result.u = (s << 63) + (e << 52) + 0x1;
            return result.f;
        }

        /* x * Inf + y = Inf */
        di_type result;
        e = 0x7ff;
        result.u = (s << 63) + (e << 52) + 0;
        return result.f;
    }

    if (c_flt_e == 0x7ff) {
        if (c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        /* x * y + Inf = Inf */
        return c;
    }

    if (a_flt_e == 0) {
        if (a_flt_m == 0) {
            /* 'a' is zero, return 'c' */
            return c;
        }
        _mesa_norm_subnormal_mantissa_f64(a_flt_m , &a_flt_e, &a_flt_m);
    }

    if (b_flt_e == 0) {
        if (b_flt_m == 0) {
            /* 'b' is zero, return 'c' */
            return c;
        }
        _mesa_norm_subnormal_mantissa_f64(b_flt_m , &b_flt_e, &b_flt_m);
    }

    e = a_flt_e + b_flt_e - 0x3fe;
    a_flt_m = (a_flt_m | 0x0010000000000000) << 10;
    b_flt_m = (b_flt_m | 0x0010000000000000) << 11;

    uint32_t m_128[4];
    _mesa_softfloat_mul_f64_to_f128_m(a_flt_m, b_flt_m, m_128);

    m = (uint64_t) m_128[index_word(4, 3)] << 32 | m_128[index_word(4, 2)];

    int64_t shift_dist = 0;
    if (!(m & 0x4000000000000000)) {
        --e;
        shift_dist = -1;
    }

    if (c_flt_e == 0) {
        if (c_flt_m == 0) {
            /* 'c' is zero, return 'a * b' */
            if (shift_dist)
                m <<= 1;

            if (m_128[index_word(4, 1)] || m_128[index_word(4, 0)])
                m |= 1;
            return _mesa_roundtozero_f64(s, e - 1, m);
        }
        _mesa_norm_subnormal_mantissa_f64(c_flt_m , &c_flt_e, &c_flt_m);
    }
    c_flt_m = (c_flt_m | 0x0010000000000000) << 10;

    uint32_t c_flt_m_128[4];
    int64_t exp_diff = e - c_flt_e;
    if (exp_diff < 0) {
        e = c_flt_e;
        if ((s == c_flt_s) || (exp_diff < -1)) {
            shift_dist -= exp_diff;
            if (shift_dist) {
                m = _mesa_shift_right_jam64(m, shift_dist);
            }
        } else {
            if (!shift_dist) {
                _mesa_short_shift_right_m(4, m_128, 1, m_128);
            }
        }
    } else {
        if (shift_dist)
            _mesa_add_m(4, m_128, m_128, m_128);
        if (!exp_diff) {
            m = (uint64_t) m_128[index_word(4, 3)] << 32
                | m_128[index_word(4, 2)];
        } else {
            c_flt_m_128[index_word(4, 3)] = c_flt_m >> 32;
            c_flt_m_128[index_word(4, 2)] = c_flt_m;
            c_flt_m_128[index_word(4, 1)] = 0;
            c_flt_m_128[index_word(4, 0)] = 0;
            _mesa_shift_right_jam_m(4, c_flt_m_128, exp_diff, c_flt_m_128);
        }
    }

    if (s == c_flt_s) {
        if (exp_diff <= 0) {
            m += c_flt_m;
        } else {
            _mesa_add_m(4, m_128, c_flt_m_128, m_128);
            m = (uint64_t) m_128[index_word(4, 3)] << 32
                | m_128[index_word(4, 2)];
        }
        if (m & 0x8000000000000000) {
            e++;
            m = _mesa_short_shift_right_jam64(m, 1);
        }
    } else {
        if (exp_diff < 0) {
            s = c_flt_s;
            if (exp_diff < -1) {
                m = c_flt_m - m;
                if (m_128[index_word(4, 1)] || m_128[index_word(4, 0)]) {
                    m = (m - 1) | 1;
                }
                if (!(m & 0x4000000000000000)) {
                    --e;
                    m <<= 1;
                }
                return _mesa_roundtozero_f64(s, e - 1, m);
            } else {
                c_flt_m_128[index_word(4, 3)] = c_flt_m >> 32;
                c_flt_m_128[index_word(4, 2)] = c_flt_m;
                c_flt_m_128[index_word(4, 1)] = 0;
                c_flt_m_128[index_word(4, 0)] = 0;
                _mesa_sub_m(4, c_flt_m_128, m_128, m_128);
            }
        } else if (!exp_diff) {
            m -= c_flt_m;
            if (!m && !m_128[index_word(4, 1)] && !m_128[index_word(4, 0)]) {
                /* Return zero */
                di_type result;
                result.u = (s << 63) + 0;
                return result.f;
            }
            m_128[index_word(4, 3)] = m >> 32;
            m_128[index_word(4, 2)] = m;
            if (m & 0x8000000000000000) {
                s = !s;
                _mesa_neg_x_m(4, m_128);
            }
        } else {
            _mesa_sub_m(4, m_128, c_flt_m_128, m_128);
            if (1 < exp_diff) {
                m = (uint64_t) m_128[index_word(4, 3)] << 32
                    | m_128[index_word(4, 2)];
                if (!(m & 0x4000000000000000)) {
                    --e;
                    m <<= 1;
                }
                if (m_128[index_word(4, 1)] || m_128[index_word(4, 0)])
                    m |= 1;
                return _mesa_roundtozero_f64(s, e - 1, m);
            }
        }

        shift_dist = 0;
        m = (uint64_t) m_128[index_word(4, 3)] << 32
            | m_128[index_word(4, 2)];
        if (!m) {
            shift_dist = 64;
            m = (uint64_t) m_128[index_word(4, 1)] << 32
                | m_128[index_word(4, 0)];
        }
        shift_dist += _mesa_count_leading_zeros64(m) - 1;
        if (shift_dist) {
            e -= shift_dist;
            _mesa_shift_left_m(4, m_128, shift_dist, m_128);
            m = (uint64_t) m_128[index_word(4, 3)] << 32
                | m_128[index_word(4, 2)];
        }
    }

    if (m_128[index_word(4, 1)] || m_128[index_word(4, 0)])
        m |= 1;
    return _mesa_roundtozero_f64(s, e - 1, m);
}


/**
 * \brief Calculate a * b + c but rounding to zero.
 *
 * Notice that this mainly differs from the original Berkeley SoftFloat 3e
 * implementation in that we don't really treat NaNs, Zeroes nor the
 * signalling flags. Any NaN is good for us and the sign of the Zero is not
 * important.
 *
 * From f32_mulAdd()
 */
float
_mesa_float_fma_rtz(float a, float b, float c)
{
    const fi_type a_fi = {a};
    uint32_t a_flt_m = a_fi.u & 0x07fffff;
    uint32_t a_flt_e = (a_fi.u >> 23) & 0xff;
    uint32_t a_flt_s = (a_fi.u >> 31) & 0x1;
    const fi_type b_fi = {b};
    uint32_t b_flt_m = b_fi.u & 0x07fffff;
    uint32_t b_flt_e = (b_fi.u >> 23) & 0xff;
    uint32_t b_flt_s = (b_fi.u >> 31) & 0x1;
    const fi_type c_fi = {c};
    uint32_t c_flt_m = c_fi.u & 0x07fffff;
    uint32_t c_flt_e = (c_fi.u >> 23) & 0xff;
    uint32_t c_flt_s = (c_fi.u >> 31) & 0x1;
    int32_t s, e, m = 0;

    c_flt_s ^= 0;
    s = a_flt_s ^ b_flt_s ^ 0;

    if (a_flt_e == 0xff) {
        if (a_flt_m != 0) {
            /* 'a' is a NaN, return NaN */
            return a;
        } else if (b_flt_e == 0xff && b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        } else if (c_flt_e == 0xff && c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        if (!(b_flt_e | b_flt_m)) {
            /* Inf * 0 + y = NaN */
            fi_type result;
            e = 0xff;
            result.u = (s << 31) + (e << 23) + 0x1;
            return result.f;
        }

        if ((c_flt_e == 0xff && c_flt_m == 0) && (s != c_flt_s)) {
            /* Inf * x - Inf = NaN */
            fi_type result;
            e = 0xff;
            result.u = (s << 31) + (e << 23) + 0x1;
            return result.f;
        }

        /* Inf * x + y = Inf */
        fi_type result;
        e = 0xff;
        result.u = (s << 31) + (e << 23) + 0;
        return result.f;
    }

    if (b_flt_e == 0xff) {
        if (b_flt_m != 0) {
            /* 'b' is a NaN, return NaN */
            return b;
        } else if (c_flt_e == 0xff && c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        if (!(a_flt_e | a_flt_m)) {
            /* 0 * Inf + y = NaN */
            fi_type result;
            e = 0xff;
            result.u = (s << 31) + (e << 23) + 0x1;
            return result.f;
        }

        if ((c_flt_e == 0xff && c_flt_m == 0) && (s != c_flt_s)) {
            /* x * Inf - Inf = NaN */
            fi_type result;
            e = 0xff;
            result.u = (s << 31) + (e << 23) + 0x1;
            return result.f;
        }

        /* x * Inf + y = Inf */
        fi_type result;
        e = 0xff;
        result.u = (s << 31) + (e << 23) + 0;
        return result.f;
    }

    if (c_flt_e == 0xff) {
        if (c_flt_m != 0) {
            /* 'c' is a NaN, return NaN */
            return c;
        }

        /* x * y + Inf = Inf */
        return c;
    }

    if (a_flt_e == 0) {
        if (a_flt_m == 0) {
            /* 'a' is zero, return 'c' */
            return c;
        }
        _mesa_norm_subnormal_mantissa_f32(a_flt_m , &a_flt_e, &a_flt_m);
    }

    if (b_flt_e == 0) {
        if (b_flt_m == 0) {
            /* 'b' is zero, return 'c' */
            return c;
        }
        _mesa_norm_subnormal_mantissa_f32(b_flt_m , &b_flt_e, &b_flt_m);
    }

    e = a_flt_e + b_flt_e - 0x7e;
    a_flt_m = (a_flt_m | 0x00800000) << 7;
    b_flt_m = (b_flt_m | 0x00800000) << 7;

    uint64_t m_64 = (uint64_t) a_flt_m * b_flt_m;
    if (m_64 < 0x2000000000000000) {
        --e;
        m_64 <<= 1;
    }

    if (c_flt_e == 0) {
        if (c_flt_m == 0) {
            /* 'c' is zero, return 'a * b' */
            m = _mesa_short_shift_right_jam64(m_64, 31);
            return _mesa_round_f32(s, e - 1, m, true);
        }
        _mesa_norm_subnormal_mantissa_f32(c_flt_m , &c_flt_e, &c_flt_m);
    }
    c_flt_m = (c_flt_m | 0x00800000) << 6;

    int16_t exp_diff = e - c_flt_e;
    if (s == c_flt_s) {
        if (exp_diff <= 0) {
            e = c_flt_e;
            m = c_flt_m + _mesa_shift_right_jam64(m_64, 32 - exp_diff);
        } else {
            m_64 += _mesa_shift_right_jam64((uint64_t) c_flt_m << 32, exp_diff);
            m = _mesa_short_shift_right_jam64(m_64, 32);
        }
        if (m < 0x40000000) {
            --e;
            m <<= 1;
        }
    } else {
        uint64_t c_flt_m_64 = (uint64_t) c_flt_m << 32;
        if (exp_diff < 0) {
            s = c_flt_s;
            e = c_flt_e;
            m_64 = c_flt_m_64 - _mesa_shift_right_jam64(m_64, -exp_diff);
        } else if (!exp_diff) {
            m_64 -= c_flt_m_64;
            if (!m_64) {
                /* Return zero */
                fi_type result;
                result.u = (s << 31) + 0;
                return result.f;
            }
            if (m_64 & 0x8000000000000000) {
                s = !s;
                m_64 = -m_64;
            }
        } else {
            m_64 -= _mesa_shift_right_jam64(c_flt_m_64, exp_diff);
        }
        int8_t shift_dist = _mesa_count_leading_zeros64(m_64) - 1;
        e -= shift_dist;
        shift_dist -= 32;
        if (shift_dist < 0) {
            m = _mesa_short_shift_right_jam64(m_64, -shift_dist);
        } else {
            m = (uint32_t) m_64 << shift_dist;
        }
    }

    return _mesa_round_f32(s, e, m, true);
}


/**
 * \brief Converts from 64bits to 32bits float and rounds according to
 * instructed.
 *
 * From f64_to_f32()
 */
float
_mesa_double_to_f32(double val, bool rtz)
{
    const di_type di = {val};
    uint64_t flt_m = di.u & 0x0fffffffffffff;
    uint64_t flt_e = (di.u >> 52) & 0x7ff;
    uint64_t flt_s = (di.u >> 63) & 0x1;
    int32_t s, e, m = 0;

    s = flt_s;

    if (flt_e == 0x7ff) {
        if (flt_m != 0) {
            /* 'val' is a NaN, return NaN */
            fi_type result;
            e = 0xff;
            m = 0x1;
            result.u = (s << 31) + (e << 23) + m;
            return result.f;
        }

        /* 'val' is Inf, return Inf */
        fi_type result;
        e = 0xff;
        result.u = (s << 31) + (e << 23) + m;
        return result.f;
    }

    if (!(flt_e | flt_m)) {
        /* 'val' is zero, return zero */
        fi_type result;
        e = 0;
        result.u = (s << 31) + (e << 23) + m;
        return result.f;
    }

    m = _mesa_short_shift_right_jam64(flt_m, 22);
    if ( ! (flt_e | m) ) {
        /* 'val' is denorm, return zero */
        fi_type result;
        e = 0;
        result.u = (s << 31) + (e << 23) + m;
        return result.f;
    }

    return _mesa_round_f32(s, flt_e - 0x381, m | 0x40000000, rtz);
}


/**
 * \brief Converts from 32bits to 16bits float and rounds the result to zero.
 *
 * From f32_to_f16()
 */
uint16_t
_mesa_float_to_half_rtz_slow(float val)
{
    const fi_type fi = {val};
    const uint32_t flt_m = fi.u & 0x7fffff;
    const uint32_t flt_e = (fi.u >> 23) & 0xff;
    const uint32_t flt_s = (fi.u >> 31) & 0x1;
    int16_t s, e, m = 0;

    s = flt_s;

    if (flt_e == 0xff) {
        if (flt_m != 0) {
            /* 'val' is a NaN, return NaN */
            e = 0x1f;
            /* Retain the top bits of a NaN to make sure that the quiet/signaling
            * status stays the same.
            */
            m = flt_m >> 13;
            if (!m)
               m = 1;
            return (s << 15) + (e << 10) + m;
        }

        /* 'val' is Inf, return Inf */
        e = 0x1f;
        return (s << 15) + (e << 10) + m;
    }

    if (!(flt_e | flt_m)) {
        /* 'val' is zero, return zero */
        e = 0;
        return (s << 15) + (e << 10) + m;
    }

    m = flt_m >> 9 | ((flt_m & 0x1ff) != 0);
    if ( ! (flt_e | m) ) {
        /* 'val' is denorm, return zero */
        e = 0;
        return (s << 15) + (e << 10) + m;
    }

    return _mesa_roundtozero_f16(s, flt_e - 0x71, m | 0x4000);
}
