#if defined(X86_SSE41) && !defined(WITHOUT_CHORBA_SSE)

#include "zbuild.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"
#include "crc32.h"
#include <emmintrin.h>
#include <smmintrin.h>
#include "arch/x86/x86_intrins.h"
#include "arch_functions.h"

#define READ_NEXT(in, off, a, b) do { \
        a = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t))); \
        b = _mm_load_si128((__m128i*)(in + off / sizeof(uint64_t) + 2)); \
        } while (0);

#define NEXT_ROUND(invec, a, b, c, d) do { \
        a = _mm_xor_si128(_mm_slli_epi64(invec, 17), _mm_slli_epi64(invec, 55)); \
        b = _mm_xor_si128(_mm_xor_si128(_mm_srli_epi64(invec, 47), _mm_srli_epi64(invec, 9)), _mm_slli_epi64(invec, 19)); \
        c = _mm_xor_si128(_mm_srli_epi64(invec, 45), _mm_slli_epi64(invec, 44)); \
        d  = _mm_srli_epi64(invec, 20); \
        } while (0);

#define REALIGN_CHORBA(in0, in1, in2, in3, out0, out1, out2, out3, out4, shift) do { \
        out0 = _mm_slli_si128(in0, shift); \
        out1 = _mm_alignr_epi8(in1, in0, shift); \
        out2 = _mm_alignr_epi8(in2, in1, shift); \
        out3 = _mm_alignr_epi8(in3, in2, shift); \
        out4 = _mm_srli_si128(in3, shift); \
        } while (0)

#define STORE4(out0, out1, out2, out3, out) do { \
        _mm_store_si128(out++, out0); \
        _mm_store_si128(out++, out1); \
        _mm_store_si128(out++, out2); \
        _mm_store_si128(out++, out3); \
    } while (0)

#define READ4(out0, out1, out2, out3, in) do { \
    out0 = _mm_load_si128(in++); \
    out1 = _mm_load_si128(in++); \
    out2 = _mm_load_si128(in++); \
    out3 = _mm_load_si128(in++); \
    } while (0)

/* This is intentionally shifted one down to compensate for the deferred store from
 * the last iteration */
#define READ4_WITHXOR(out0, out1, out2, out3, xor0, xor1, xor2, xor3, in) do { \
    out0 = _mm_xor_si128(in[1], xor0); \
    out1 = _mm_xor_si128(in[2], xor1); \
    out2 = _mm_xor_si128(in[3], xor2); \
    out3 = _mm_xor_si128(in[4], xor3); \
    } while (0)

static Z_FORCEINLINE uint32_t crc32_chorba_32768_nondestructive_sse41(uint32_t crc, const uint64_t* buf, size_t len) {
    const uint64_t* input = buf;
    ALIGNED_(16) uint64_t bitbuffer[32768 / sizeof(uint64_t)];
    __m128i *bitbuffer_v = (__m128i*)bitbuffer;
    const uint8_t* bitbufferbytes = (const uint8_t*) bitbuffer;
    __m128i z = _mm_setzero_si128();

    __m128i *bitbuf128 = &bitbuffer_v[64];
    __m128i *bitbuf144 = &bitbuffer_v[72];
    __m128i *bitbuf182 = &bitbuffer_v[91];
    __m128i *bitbuf210 = &bitbuffer_v[105];
    __m128i *bitbuf300 = &bitbuffer_v[150];
    __m128i *bitbuf0 = bitbuf128;
    __m128i *inptr = (__m128i*)input;

    /* We only need to zero out the bytes between the 128'th value and the 144th
     * that are actually read */
    __m128i *z_cursor = bitbuf128;
    for (size_t i = 0; i < 2; ++i) {
        STORE4(z, z, z, z, z_cursor);
    }

    /* We only need to zero out the bytes between the 144'th value and the 182nd that
     * are actually read */
    z_cursor = bitbuf144 + 8;
    for (size_t i = 0; i < 11; ++i) {
        _mm_store_si128(z_cursor++, z);
    }

    /* We only need to zero out the bytes between the 182nd value and the 210th that
     * are actually read. */
    z_cursor = bitbuf182;
    for (size_t i = 0; i < 4; ++i) {
        STORE4(z, z, z, z, z_cursor);
    }

    /* We need to mix this in */
    __m128i init_crc = _mm_cvtsi64_si128(crc);
    crc = 0;

    size_t i = 0;

    /* Previous iteration runs carried over */
    __m128i buf144 = z;
    __m128i buf182 = z;
    __m128i buf210 = z;

    for(; i + 300*8+64 < len && i < 22 * 8; i += 64) {
        __m128i in12, in34, in56, in78,
                in_1, in23, in45, in67, in8_;

        READ4(in12, in34, in56, in78, inptr);

        if (i == 0) {
            in12 = _mm_xor_si128(in12, init_crc);
        }

        REALIGN_CHORBA(in12, in34, in56, in78,
                       in_1, in23, in45, in67, in8_, 8);

        __m128i a = _mm_xor_si128(buf144, in_1);

        STORE4(a, in23, in45, in67, bitbuf144);
        buf144 = in8_;

        __m128i e = _mm_xor_si128(buf182, in_1);
        STORE4(e, in23, in45, in67, bitbuf182);
        buf182 = in8_;

        __m128i m = _mm_xor_si128(buf210, in_1);
        STORE4(m, in23, in45, in67, bitbuf210);
        buf210 = in8_;

        STORE4(in12, in34, in56, in78, bitbuf300);
    }

    for(; i + 300*8+64 < len && i < 32 * 8; i += 64) {
        __m128i in12, in34, in56, in78,
                in_1, in23, in45, in67, in8_;
        READ4(in12, in34, in56, in78, inptr);

        REALIGN_CHORBA(in12, in34, in56, in78,
                       in_1, in23, in45, in67, in8_, 8);

        __m128i a = _mm_xor_si128(buf144, in_1);

        STORE4(a, in23, in45, in67, bitbuf144);
        buf144 = in8_;

        __m128i e, f, g, h;
        e = _mm_xor_si128(buf182, in_1);
        READ4_WITHXOR(f, g, h, buf182, in23, in45, in67, in8_, bitbuf182);
        STORE4(e, f, g, h, bitbuf182);

        __m128i m = _mm_xor_si128(buf210, in_1);
        STORE4(m, in23, in45, in67, bitbuf210);
        buf210 = in8_;

        STORE4(in12, in34, in56, in78, bitbuf300);
    }

    for(; i + 300*8+64 < len && i < 84 * 8; i += 64) {
        __m128i in12, in34, in56, in78,
                in_1, in23, in45, in67, in8_;
        READ4(in12, in34, in56, in78, inptr);

        REALIGN_CHORBA(in12, in34, in56, in78,
                       in_1, in23, in45, in67, in8_, 8);

        __m128i a, b, c, d;
        a = _mm_xor_si128(buf144, in_1);
        READ4_WITHXOR(b, c, d, buf144, in23, in45, in67, in8_, bitbuf144);
        STORE4(a, b, c, d, bitbuf144);

        __m128i e, f, g, h;
        e = _mm_xor_si128(buf182, in_1);
        READ4_WITHXOR(f, g, h, buf182, in23, in45, in67, in8_, bitbuf182);
        STORE4(e, f, g, h, bitbuf182);

        __m128i m = _mm_xor_si128(buf210, in_1);
        STORE4(m, in23, in45, in67, bitbuf210);
        buf210 = in8_;

        STORE4(in12, in34, in56, in78, bitbuf300);
    }

    for(; i + 300*8+64 < len; i += 64) {
        __m128i in12, in34, in56, in78,
                in_1, in23, in45, in67, in8_;

        if (i < 128 * 8) {
            READ4(in12, in34, in56, in78, inptr);
        } else {
            in12 = _mm_xor_si128(_mm_load_si128(inptr++), _mm_load_si128(bitbuf0++));
            in34 = _mm_xor_si128(_mm_load_si128(inptr++), _mm_load_si128(bitbuf0++));
            in56 = _mm_xor_si128(_mm_load_si128(inptr++), _mm_load_si128(bitbuf0++));
            in78 = _mm_xor_si128(_mm_load_si128(inptr++), _mm_load_si128(bitbuf0++));
        }

        // [0, 145, 183, 211]

        /* Pre Penryn CPUs the unpack should be faster */
        REALIGN_CHORBA(in12, in34, in56, in78,
                       in_1, in23, in45, in67, in8_, 8);

        __m128i a, b, c, d;
        a = _mm_xor_si128(buf144, in_1);
        READ4_WITHXOR(b, c, d, buf144, in23, in45, in67, in8_, bitbuf144);
        STORE4(a, b, c, d, bitbuf144);

        __m128i e, f, g, h;
        e = _mm_xor_si128(buf182, in_1);
        READ4_WITHXOR(f, g, h, buf182, in23, in45, in67, in8_, bitbuf182);
        STORE4(e, f, g, h, bitbuf182);

        __m128i n, o, p;
        __m128i m = _mm_xor_si128(buf210, in_1);

        /* Couldn't tell you why but despite knowing that this is always false,
         * removing this branch with GCC makes things significantly slower. Some
         * loop bodies must be being joined or something */
        if (i < 84 * 8) {
            n = in23;
            o = in45;
            p = in67;
            buf210 = in8_;
        } else {
            READ4_WITHXOR(n, o, p, buf210, in23, in45, in67, in8_, bitbuf210);
        }

        STORE4(m, n, o, p, bitbuf210);
        STORE4(in12, in34, in56, in78, bitbuf300);
    }

    /* Second half of stores bubbled out */
    _mm_store_si128(bitbuf144, buf144);
    _mm_store_si128(bitbuf182, buf182);
    _mm_store_si128(bitbuf210, buf210);

    /* We also have to zero out the tail */
    size_t left_to_z = len - (300*8 + i);
    __m128i *bitbuf_tail = (__m128i*)(bitbuffer + 300 + i/8);
    while (left_to_z >= 64) {
       STORE4(z, z, z, z, bitbuf_tail);
       left_to_z -= 64;
    }

    while (left_to_z >= 16) {
       _mm_store_si128(bitbuf_tail++, z);
       left_to_z -= 16;
    }

    uint8_t *tail_bytes = (uint8_t*)bitbuf_tail;
    while (left_to_z--) {
       *tail_bytes++ = 0;
    }

    ALIGNED_(16) uint64_t final[9] = {0};
    __m128i next12, next34, next56;
    next12 = z;
    next34 = z;
    next56 = z;

    for(; (i + 72 < len); i += 32) {
        __m128i in1in2, in3in4;
        __m128i in1in2_, in3in4_;
        __m128i ab1, ab2, ab3, ab4;
        __m128i cd1, cd2, cd3, cd4;

        READ_NEXT(input, i, in1in2, in3in4);
        READ_NEXT(bitbuffer, i, in1in2_, in3in4_);

        in1in2 = _mm_xor_si128(_mm_xor_si128(in1in2, in1in2_), next12);
        in3in4 = _mm_xor_si128(in3in4, in3in4_);

        NEXT_ROUND(in1in2, ab1, ab2, ab3, ab4);

        __m128i a2_ = _mm_slli_si128(ab2, 8);
        __m128i ab1_next34 = _mm_xor_si128(next34, ab1);
        in3in4 = _mm_xor_si128(in3in4, ab1_next34);
        in3in4 = _mm_xor_si128(a2_, in3in4);
        NEXT_ROUND(in3in4, cd1, cd2, cd3, cd4);

        __m128i b2c2 = _mm_alignr_epi8(cd2, ab2, 8);
        __m128i a4_ = _mm_slli_si128(ab4, 8);
        a4_ = _mm_xor_si128(b2c2, a4_);
        next12 = _mm_xor_si128(ab3, a4_);
        next12 = _mm_xor_si128(next12, cd1);

        __m128i d2_ = _mm_srli_si128(cd2, 8);
        __m128i b4c4 = _mm_alignr_epi8(cd4, ab4, 8);
        next12 = _mm_xor_si128(next12, next56);
        next34 = _mm_xor_si128(cd3, _mm_xor_si128(b4c4, d2_));
        next56 = _mm_srli_si128(cd4, 8);
    }

    memcpy(final, input+(i / sizeof(uint64_t)), len-i);
    __m128i *final128 = (__m128i*)final;
    _mm_store_si128(final128, _mm_xor_si128(_mm_load_si128(final128), next12));
    ++final128;
    _mm_store_si128(final128, _mm_xor_si128(_mm_load_si128(final128), next34));
    ++final128;
    _mm_store_si128(final128, _mm_xor_si128(_mm_load_si128(final128), next56));

    uint8_t* final_bytes = (uint8_t*) final;

    for(size_t j = 0; j < (len-i); j++) {
        crc = crc_table[(crc ^ final_bytes[j] ^ bitbufferbytes[(j+i)]) & 0xff] ^ (crc >> 8);
    }
    return crc;
}

Z_INTERNAL uint32_t crc32_chorba_sse41(uint32_t crc, const uint8_t *buf, size_t len) {
    uint64_t* aligned_buf;
    uint32_t c = (~crc) & 0xffffffff;
    uintptr_t algn_diff = ((uintptr_t)16 - ((uintptr_t)buf & 15)) & 15;

    if (len > algn_diff + CHORBA_SMALL_THRESHOLD_64BIT) {
        if (algn_diff) {
            c = crc32_braid_internal(c, buf, algn_diff);
            len -= algn_diff;
        }
        aligned_buf = (uint64_t*) (buf + algn_diff);
#if !defined(WITHOUT_CHORBA)
        if(len > CHORBA_LARGE_THRESHOLD) {
            c = crc32_chorba_118960_nondestructive(c, (z_word_t*) aligned_buf, len);
        } else
#endif
        if (len > CHORBA_MEDIUM_LOWER_THRESHOLD && len <= CHORBA_MEDIUM_UPPER_THRESHOLD) {
            c = crc32_chorba_32768_nondestructive_sse41(c, aligned_buf, len);
        } else {
            c = chorba_small_nondestructive_sse2(c, aligned_buf, len);
        }
    } else {
        // Process too short lengths using crc32_braid
        c = crc32_braid_internal(c, buf, len);
    }

    /* Return the CRC, post-conditioned. */
    return c ^ 0xffffffff;
}
#endif
