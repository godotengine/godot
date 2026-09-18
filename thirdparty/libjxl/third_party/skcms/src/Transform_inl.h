/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

// Intentionally NO #pragma once... included multiple times.

// This file is included from skcms.cc in a namespace with some pre-defines:
//    - N:    SIMD width of all vectors; 1, 4, 8 or 16 (preprocessor define)
//    - V<T>: a template to create a vector of N T's.

using F   = V<float>;
using I32 = V<int32_t>;
using U64 = V<uint64_t>;
using U32 = V<uint32_t>;
using U16 = V<uint16_t>;
using U8  = V<uint8_t>;

#if defined(__GNUC__) && !defined(__clang__)
    // GCC is kind of weird, not allowing vector = scalar directly.
    static constexpr F F0 = F() + 0.0f,
                       F1 = F() + 1.0f,
                       FInfBits = F() + 0x7f800000; // equals 2139095040, the bit pattern of +Inf
#else
    static constexpr F F0 = 0.0f,
                       F1 = 1.0f,
                       FInfBits = 0x7f800000; // equals 2139095040, the bit pattern of +Inf
#endif

// Instead of checking __AVX__ below, we'll check USING_AVX.
// This lets skcms.cc set USING_AVX to force us in even if the compiler's not set that way.
// Same deal for __F16C__ and __AVX2__ ~~~> USING_AVX_F16C, USING_AVX2.

#if !defined(USING_AVX)      && N == 8 && defined(__AVX__)
    #define  USING_AVX
#endif
#if !defined(USING_AVX_F16C) && defined(USING_AVX) && defined(__F16C__)
    #define  USING_AVX_F16C
#endif
#if !defined(USING_AVX2)     && defined(USING_AVX) && defined(__AVX2__)
    #define  USING_AVX2
#endif
#if !defined(USING_AVX512F)  && N == 16 && defined(__AVX512F__) && defined(__AVX512DQ__)
    #define  USING_AVX512F
#endif

// Similar to the AVX+ features, we define USING_NEON and USING_NEON_F16C.
// This is more for organizational clarity... skcms.cc doesn't force these.
#if N > 1 && defined(__ARM_NEON)
    #define USING_NEON

    // We have to use two different mechanisms to enable the f16 conversion intrinsics:
    #if defined(__clang__)
        // Clang's arm_neon.h guards them with the FP hardware bit:
        #if __ARM_FP & 2
            #define USING_NEON_F16C
        #endif
    #elif defined(__GNUC__)
        // GCC's arm_neon.h guards them with the FP16 format macros (IEEE and ALTERNATIVE).
        // We don't actually want the alternative format - we're reading/writing IEEE f16 values.
        #if defined(__ARM_FP16_FORMAT_IEEE)
            #define USING_NEON_F16C
        #endif
    #endif
#endif

// These -Wvector-conversion warnings seem to trigger in very bogus situations,
// like vst3q_f32() expecting a 16x char rather than a 4x float vector.  :/
#if defined(USING_NEON) && defined(__clang__)
    #pragma clang diagnostic ignored "-Wvector-conversion"
#endif

// GCC & Clang (but not clang-cl) warn returning U64 on x86 is larger than a register.
// You'd see warnings like, "using AVX even though AVX is not enabled".
// We stifle these warnings; our helpers that return U64 are always inlined.
#if defined(__SSE__) && defined(__GNUC__)
    #if !defined(__has_warning)
        #pragma GCC diagnostic ignored "-Wpsabi"
    #elif __has_warning("-Wpsabi")
        #pragma GCC diagnostic ignored "-Wpsabi"
    #endif
#endif

// We tag most helper functions as SI, to enforce good code generation
// but also work around what we think is a bug in GCC: when targeting 32-bit
// x86, GCC tends to pass U16 (4x uint16_t vector) function arguments in the
// MMX mm0 register, which seems to mess with unrelated code that later uses
// x87 FP instructions (MMX's mm0 is an alias for x87's st0 register).
#if defined(__clang__) || defined(__GNUC__)
    #define SI static inline __attribute__((always_inline))
#else
    #define SI static inline
#endif

template <typename T, typename P>
SI T load(const P* ptr) {
    T val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}
template <typename T, typename P>
SI void store(P* ptr, const T& val) {
    memcpy(ptr, &val, sizeof(val));
}

// (T)v is a cast when N == 1 and a bit-pun when N>1,
// so we use cast<T>(v) to actually cast or bit_pun<T>(v) to bit-pun.
template <typename D, typename S>
SI D cast(const S& v) {
#if N == 1
    return (D)v;
#elif defined(__clang__)
    return __builtin_convertvector(v, D);
#else
    D d;
    for (int i = 0; i < N; i++) {
        d[i] = v[i];
    }
    return d;
#endif
}

template <typename D, typename S>
SI D bit_pun(const S& v) {
    static_assert(sizeof(D) == sizeof(v), "");
    return load<D>(&v);
}

// When we convert from float to fixed point, it's very common to want to round,
// and for some reason compilers generate better code when converting to int32_t.
// To serve both those ends, we use this function to_fixed() instead of direct cast().
SI U32 to_fixed(F f) {  return (U32)cast<I32>(f + 0.5f); }

// Sometimes we do something crazy on one branch of a conditonal,
// like divide by zero or convert a huge float to an integer,
// but then harmlessly select the other side.  That trips up N==1
// sanitizer builds, so we make if_then_else() a macro to avoid
// evaluating the unused side.

#if N == 1
    #define if_then_else(cond, t, e) ((cond) ? (t) : (e))
#else
    template <typename C, typename T>
    SI T if_then_else(C cond, T t, T e) {
        return bit_pun<T>( ( cond & bit_pun<C>(t)) |
                           (~cond & bit_pun<C>(e)) );
    }
#endif


SI F F_from_Half(U16 half) {
#if defined(USING_NEON_F16C)
    return vcvt_f32_f16((float16x4_t)half);
#elif defined(USING_AVX512F)
    return (F)_mm512_cvtph_ps((__m256i)half);
#elif defined(USING_AVX_F16C)
    typedef int16_t __attribute__((vector_size(16))) I16;
    return __builtin_ia32_vcvtph2ps256((I16)half);
#else
    U32 wide = cast<U32>(half);
    // A half is 1-5-10 sign-exponent-mantissa, with 15 exponent bias.
    U32 s  = wide & 0x8000,
        em = wide ^ s;

    // Constructing the float is easy if the half is not denormalized.
    F norm = bit_pun<F>( (s<<16) + (em<<13) + ((127-15)<<23) );

    // Simply flush all denorm half floats to zero.
    return if_then_else(em < 0x0400, F0, norm);
#endif
}

#if defined(__clang__)
    // The -((127-15)<<10) underflows that side of the math when
    // we pass a denorm half float.  It's harmless... we'll take the 0 side anyway.
    __attribute__((no_sanitize("unsigned-integer-overflow")))
#endif
SI U16 Half_from_F(F f) {
#if defined(USING_NEON_F16C)
    return (U16)vcvt_f16_f32(f);
#elif defined(USING_AVX512F)
    return (U16)_mm512_cvtps_ph((__m512 )f, _MM_FROUND_CUR_DIRECTION );
#elif defined(USING_AVX_F16C)
    return (U16)__builtin_ia32_vcvtps2ph256(f, 0x04/*_MM_FROUND_CUR_DIRECTION*/);
#else
    // A float is 1-8-23 sign-exponent-mantissa, with 127 exponent bias.
    U32 sem = bit_pun<U32>(f),
        s   = sem & 0x80000000,
         em = sem ^ s;

    // For simplicity we flush denorm half floats (including all denorm floats) to zero.
    return cast<U16>(if_then_else(em < 0x38800000, (U32)F0
                                                 , (s>>16) + (em>>13) - ((127-15)<<10)));
#endif
}

// Swap high and low bytes of 16-bit lanes, converting between big-endian and little-endian.
#if defined(USING_NEON)
    SI U16 swap_endian_16(U16 v) {
        return (U16)vrev16_u8((uint8x8_t) v);
    }
#endif

SI U64 swap_endian_16x4(const U64& rgba) {
    return (rgba & 0x00ff00ff00ff00ff) << 8
         | (rgba & 0xff00ff00ff00ff00) >> 8;
}

#if defined(USING_NEON)
    SI F min_(F x, F y) { return (F)vminq_f32((float32x4_t)x, (float32x4_t)y); }
    SI F max_(F x, F y) { return (F)vmaxq_f32((float32x4_t)x, (float32x4_t)y); }
#elif defined(__loongarch_sx)
    SI F min_(F x, F y) { return (F)__lsx_vfmin_s(x, y); }
    SI F max_(F x, F y) { return (F)__lsx_vfmax_s(x, y); }
#else
    SI F min_(F x, F y) { return if_then_else(x > y, y, x); }
    SI F max_(F x, F y) { return if_then_else(x < y, y, x); }
#endif

SI F floor_(F x) {
#if N == 1
    return floorf_(x);
#elif defined(__aarch64__)
    return vrndmq_f32(x);
#elif defined(USING_AVX512F)
    // Clang's _mm512_floor_ps() passes its mask as -1, not (__mmask16)-1,
    // and integer santizer catches that this implicit cast changes the
    // value from -1 to 65535.  We'll cast manually to work around it.
    // Read this as `return _mm512_floor_ps(x)`.
    return _mm512_mask_floor_ps(x, (__mmask16)-1, x);
#elif defined(USING_AVX)
    return __builtin_ia32_roundps256(x, 0x01/*_MM_FROUND_FLOOR*/);
#elif defined(__SSE4_1__)
    return _mm_floor_ps(x);
#elif defined(__loongarch_sx)
    return __lsx_vfrintrm_s((__m128)x);
#else
    // Round trip through integers with a truncating cast.
    F roundtrip = cast<F>(cast<I32>(x));
    // If x is negative, truncating gives the ceiling instead of the floor.
    return roundtrip - if_then_else(roundtrip > x, F1, F0);

    // This implementation fails for values of x that are outside
    // the range an integer can represent.  We expect most x to be small.
#endif
}

SI F approx_log2(F x) {
    // The first approximation of log2(x) is its exponent 'e', minus 127.
    I32 bits = bit_pun<I32>(x);

    F e = cast<F>(bits) * (1.0f / (1<<23));

    // If we use the mantissa too we can refine the error signficantly.
    F m = bit_pun<F>( (bits & 0x007fffff) | 0x3f000000 );

    return e - 124.225514990f
             -   1.498030302f*m
             -   1.725879990f/(0.3520887068f + m);
}

SI F approx_log(F x) {
    const float ln2 = 0.69314718f;
    return ln2 * approx_log2(x);
}

SI F approx_exp2(F x) {
    F fract = x - floor_(x);

    F fbits = (1.0f * (1<<23)) * (x + 121.274057500f
                                    -   1.490129070f*fract
                                    +  27.728023300f/(4.84252568f - fract));
    I32 bits = cast<I32>(min_(max_(fbits, F0), FInfBits));

    return bit_pun<F>(bits);
}

SI F approx_pow(F x, float y) {
    return if_then_else((x == F0) | (x == F1), x
                                             , approx_exp2(approx_log2(x) * y));
}

SI F approx_exp(F x) {
    const float log2_e = 1.4426950408889634074f;
    return approx_exp2(log2_e * x);
}

SI F strip_sign(F x, U32* sign) {
    U32 bits = bit_pun<U32>(x);
    *sign = bits & 0x80000000;
    return bit_pun<F>(bits ^ *sign);
}

SI F apply_sign(F x, U32 sign) {
    return bit_pun<F>(sign | bit_pun<U32>(x));
}

// Return tf(x).
SI F apply_tf(const skcms_TransferFunction* tf, F x) {
    // Peel off the sign bit and set x = |x|.
    U32 sign;
    x = strip_sign(x, &sign);

    // The transfer function has a linear part up to d, exponential at d and after.
    F v = if_then_else(x < tf->d,            tf->c*x + tf->f
                                , approx_pow(tf->a*x + tf->b, tf->g) + tf->e);

    // Tack the sign bit back on.
    return apply_sign(v, sign);
}

// Return the gamma function (|x|^G with the original sign re-applied to x).
SI F apply_gamma(const skcms_TransferFunction* tf, F x) {
    U32 sign;
    x = strip_sign(x, &sign);
    return apply_sign(approx_pow(x, tf->g), sign);
}

SI F apply_pq(const skcms_TransferFunction* tf, F x) {
    U32 bits = bit_pun<U32>(x),
        sign = bits & 0x80000000;
    x = bit_pun<F>(bits ^ sign);

    F v = approx_pow(max_(tf->a + tf->b * approx_pow(x, tf->c), F0)
                       / (tf->d + tf->e * approx_pow(x, tf->c)),
                     tf->f);

    return bit_pun<F>(sign | bit_pun<U32>(v));
}

SI F apply_hlg(const skcms_TransferFunction* tf, F x) {
    const float R = tf->a, G = tf->b,
                a = tf->c, b = tf->d, c = tf->e,
                K = tf->f + 1;
    U32 bits = bit_pun<U32>(x),
        sign = bits & 0x80000000;
    x = bit_pun<F>(bits ^ sign);

    F v = if_then_else(x*R <= 1, approx_pow(x*R, G)
                               , approx_exp((x-c)*a) + b);

    return K*bit_pun<F>(sign | bit_pun<U32>(v));
}

SI F apply_hlginv(const skcms_TransferFunction* tf, F x) {
    const float R = tf->a, G = tf->b,
                a = tf->c, b = tf->d, c = tf->e,
                K = tf->f + 1;
    U32 bits = bit_pun<U32>(x),
        sign = bits & 0x80000000;
    x = bit_pun<F>(bits ^ sign);
    x /= K;

    F v = if_then_else(x <= 1, R * approx_pow(x, G)
                             , a * approx_log(x - b) + c);

    return bit_pun<F>(sign | bit_pun<U32>(v));
}


// Strided loads and stores of N values, starting from p.
template <typename T, typename P>
SI T load_3(const P* p) {
#if N == 1
    return (T)p[0];
#elif N == 4
    return T{p[ 0],p[ 3],p[ 6],p[ 9]};
#elif N == 8
    return T{p[ 0],p[ 3],p[ 6],p[ 9], p[12],p[15],p[18],p[21]};
#elif N == 16
    return T{p[ 0],p[ 3],p[ 6],p[ 9], p[12],p[15],p[18],p[21],
             p[24],p[27],p[30],p[33], p[36],p[39],p[42],p[45]};
#endif
}

template <typename T, typename P>
SI T load_4(const P* p) {
#if N == 1
    return (T)p[0];
#elif N == 4
    return T{p[ 0],p[ 4],p[ 8],p[12]};
#elif N == 8
    return T{p[ 0],p[ 4],p[ 8],p[12], p[16],p[20],p[24],p[28]};
#elif N == 16
    return T{p[ 0],p[ 4],p[ 8],p[12], p[16],p[20],p[24],p[28],
             p[32],p[36],p[40],p[44], p[48],p[52],p[56],p[60]};
#endif
}

template <typename T, typename P>
SI void store_3(P* p, const T& v) {
#if N == 1
    p[0] = v;
#elif N == 4
    p[ 0] = v[ 0]; p[ 3] = v[ 1]; p[ 6] = v[ 2]; p[ 9] = v[ 3];
#elif N == 8
    p[ 0] = v[ 0]; p[ 3] = v[ 1]; p[ 6] = v[ 2]; p[ 9] = v[ 3];
    p[12] = v[ 4]; p[15] = v[ 5]; p[18] = v[ 6]; p[21] = v[ 7];
#elif N == 16
    p[ 0] = v[ 0]; p[ 3] = v[ 1]; p[ 6] = v[ 2]; p[ 9] = v[ 3];
    p[12] = v[ 4]; p[15] = v[ 5]; p[18] = v[ 6]; p[21] = v[ 7];
    p[24] = v[ 8]; p[27] = v[ 9]; p[30] = v[10]; p[33] = v[11];
    p[36] = v[12]; p[39] = v[13]; p[42] = v[14]; p[45] = v[15];
#endif
}

template <typename T, typename P>
SI void store_4(P* p, const T& v) {
#if N == 1
    p[0] = v;
#elif N == 4
    p[ 0] = v[ 0]; p[ 4] = v[ 1]; p[ 8] = v[ 2]; p[12] = v[ 3];
#elif N == 8
    p[ 0] = v[ 0]; p[ 4] = v[ 1]; p[ 8] = v[ 2]; p[12] = v[ 3];
    p[16] = v[ 4]; p[20] = v[ 5]; p[24] = v[ 6]; p[28] = v[ 7];
#elif N == 16
    p[ 0] = v[ 0]; p[ 4] = v[ 1]; p[ 8] = v[ 2]; p[12] = v[ 3];
    p[16] = v[ 4]; p[20] = v[ 5]; p[24] = v[ 6]; p[28] = v[ 7];
    p[32] = v[ 8]; p[36] = v[ 9]; p[40] = v[10]; p[44] = v[11];
    p[48] = v[12]; p[52] = v[13]; p[56] = v[14]; p[60] = v[15];
#endif
}


SI U8 gather_8(const uint8_t* p, I32 ix) {
#if N == 1
    U8 v = p[ix];
#elif N == 4
    U8 v = { p[ix[0]], p[ix[1]], p[ix[2]], p[ix[3]] };
#elif N == 8
    U8 v = { p[ix[0]], p[ix[1]], p[ix[2]], p[ix[3]],
             p[ix[4]], p[ix[5]], p[ix[6]], p[ix[7]] };
#elif N == 16
    U8 v = { p[ix[ 0]], p[ix[ 1]], p[ix[ 2]], p[ix[ 3]],
             p[ix[ 4]], p[ix[ 5]], p[ix[ 6]], p[ix[ 7]],
             p[ix[ 8]], p[ix[ 9]], p[ix[10]], p[ix[11]],
             p[ix[12]], p[ix[13]], p[ix[14]], p[ix[15]] };
#endif
    return v;
}

SI U16 gather_16(const uint8_t* p, I32 ix) {
    // Load the i'th 16-bit value from p.
    auto load_16 = [p](int i) {
        return load<uint16_t>(p + 2*i);
    };
#if N == 1
    U16 v = load_16(ix);
#elif N == 4
    U16 v = { load_16(ix[0]), load_16(ix[1]), load_16(ix[2]), load_16(ix[3]) };
#elif N == 8
    U16 v = { load_16(ix[0]), load_16(ix[1]), load_16(ix[2]), load_16(ix[3]),
              load_16(ix[4]), load_16(ix[5]), load_16(ix[6]), load_16(ix[7]) };
#elif N == 16
    U16 v = { load_16(ix[ 0]), load_16(ix[ 1]), load_16(ix[ 2]), load_16(ix[ 3]),
              load_16(ix[ 4]), load_16(ix[ 5]), load_16(ix[ 6]), load_16(ix[ 7]),
              load_16(ix[ 8]), load_16(ix[ 9]), load_16(ix[10]), load_16(ix[11]),
              load_16(ix[12]), load_16(ix[13]), load_16(ix[14]), load_16(ix[15]) };
#endif
    return v;
}

SI U32 gather_32(const uint8_t* p, I32 ix) {
    // Load the i'th 32-bit value from p.
    auto load_32 = [p](int i) {
        return load<uint32_t>(p + 4*i);
    };
#if N == 1
    U32 v = load_32(ix);
#elif N == 4
    U32 v = { load_32(ix[0]), load_32(ix[1]), load_32(ix[2]), load_32(ix[3]) };
#elif N == 8
    U32 v = { load_32(ix[0]), load_32(ix[1]), load_32(ix[2]), load_32(ix[3]),
              load_32(ix[4]), load_32(ix[5]), load_32(ix[6]), load_32(ix[7]) };
#elif N == 16
    U32 v = { load_32(ix[ 0]), load_32(ix[ 1]), load_32(ix[ 2]), load_32(ix[ 3]),
              load_32(ix[ 4]), load_32(ix[ 5]), load_32(ix[ 6]), load_32(ix[ 7]),
              load_32(ix[ 8]), load_32(ix[ 9]), load_32(ix[10]), load_32(ix[11]),
              load_32(ix[12]), load_32(ix[13]), load_32(ix[14]), load_32(ix[15]) };
#endif
    // TODO: AVX2 and AVX-512 gathers (c.f. gather_24).
    return v;
}

SI U32 gather_24(const uint8_t* p, I32 ix) {
    // First, back up a byte.  Any place we're gathering from has a safe junk byte to read
    // in front of it, either a previous table value, or some tag metadata.
    p -= 1;

    // Load the i'th 24-bit value from p, and 1 extra byte.
    auto load_24_32 = [p](int i) {
        return load<uint32_t>(p + 3*i);
    };

    // Now load multiples of 4 bytes (a junk byte, then r,g,b).
#if N == 1
    U32 v = load_24_32(ix);
#elif N == 4
    U32 v = { load_24_32(ix[0]), load_24_32(ix[1]), load_24_32(ix[2]), load_24_32(ix[3]) };
#elif N == 8 && !defined(USING_AVX2)
    U32 v = { load_24_32(ix[0]), load_24_32(ix[1]), load_24_32(ix[2]), load_24_32(ix[3]),
              load_24_32(ix[4]), load_24_32(ix[5]), load_24_32(ix[6]), load_24_32(ix[7]) };
#elif N == 8
    (void)load_24_32;
    // The gather instruction here doesn't need any particular alignment,
    // but the intrinsic takes a const int*.
    const int* p4 = bit_pun<const int*>(p);
    I32 zero = { 0, 0, 0, 0,  0, 0, 0, 0},
        mask = {-1,-1,-1,-1, -1,-1,-1,-1};
    #if defined(__clang__)
        U32 v = (U32)__builtin_ia32_gatherd_d256(zero, p4, 3*ix, mask, 1);
    #elif defined(__GNUC__)
        U32 v = (U32)__builtin_ia32_gathersiv8si(zero, p4, 3*ix, mask, 1);
    #endif
#elif N == 16
    (void)load_24_32;
    // The intrinsic is supposed to take const void* now, but it takes const int*, just like AVX2.
    // And AVX-512 swapped the order of arguments.  :/
    const int* p4 = bit_pun<const int*>(p);
    U32 v = (U32)_mm512_i32gather_epi32((__m512i)(3*ix), p4, 1);
#endif

    // Shift off the junk byte, leaving r,g,b in low 24 bits (and zero in the top 8).
    return v >> 8;
}

#if !defined(__arm__)
    SI void gather_48(const uint8_t* p, I32 ix, U64* v) {
        // As in gather_24(), with everything doubled.
        p -= 2;

        // Load the i'th 48-bit value from p, and 2 extra bytes.
        auto load_48_64 = [p](int i) {
            return load<uint64_t>(p + 6*i);
        };

    #if N == 1
        *v = load_48_64(ix);
    #elif N == 4
        *v = U64{
            load_48_64(ix[0]), load_48_64(ix[1]), load_48_64(ix[2]), load_48_64(ix[3]),
        };
    #elif N == 8 && !defined(USING_AVX2)
        *v = U64{
            load_48_64(ix[0]), load_48_64(ix[1]), load_48_64(ix[2]), load_48_64(ix[3]),
            load_48_64(ix[4]), load_48_64(ix[5]), load_48_64(ix[6]), load_48_64(ix[7]),
        };
    #elif N == 8
        (void)load_48_64;
        typedef int32_t   __attribute__((vector_size(16))) Half_I32;
        typedef long long __attribute__((vector_size(32))) Half_I64;

        // The gather instruction here doesn't need any particular alignment,
        // but the intrinsic takes a const long long*.
        const long long int* p8 = bit_pun<const long long int*>(p);

        Half_I64 zero = { 0, 0, 0, 0},
                 mask = {-1,-1,-1,-1};

        ix *= 6;
        Half_I32 ix_lo = { ix[0], ix[1], ix[2], ix[3] },
                 ix_hi = { ix[4], ix[5], ix[6], ix[7] };

        #if defined(__clang__)
            Half_I64 lo = (Half_I64)__builtin_ia32_gatherd_q256(zero, p8, ix_lo, mask, 1),
                     hi = (Half_I64)__builtin_ia32_gatherd_q256(zero, p8, ix_hi, mask, 1);
        #elif defined(__GNUC__)
            Half_I64 lo = (Half_I64)__builtin_ia32_gathersiv4di(zero, p8, ix_lo, mask, 1),
                     hi = (Half_I64)__builtin_ia32_gathersiv4di(zero, p8, ix_hi, mask, 1);
        #endif
        store((char*)v +  0, lo);
        store((char*)v + 32, hi);
    #elif N == 16
        (void)load_48_64;
        const long long int* p8 = bit_pun<const long long int*>(p);
        __m512i lo = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32((__m512i)(6*ix), 0), p8, 1),
                hi = _mm512_i32gather_epi64(_mm512_extracti32x8_epi32((__m512i)(6*ix), 1), p8, 1);
        store((char*)v +  0, lo);
        store((char*)v + 64, hi);
    #endif

        *v >>= 16;
    }
#endif

SI F F_from_U8(U8 v) {
    return cast<F>(v) * (1/255.0f);
}

SI F F_from_U16_BE(U16 v) {
    // All 16-bit ICC values are big-endian, so we byte swap before converting to float.
    // MSVC catches the "loss" of data here in the portable path, so we also make sure to mask.
    U16 lo = (v >> 8),
        hi = (v << 8) & 0xffff;
    return cast<F>(lo|hi) * (1/65535.0f);
}

SI U16 U16_from_F(F v) {
    // 65535 == inf in FP16, so promote to FP32 before converting.
    return cast<U16>(cast<V<float>>(v) * 65535 + 0.5f);
}

SI F minus_1_ulp(F v) {
    return bit_pun<F>( bit_pun<U32>(v) - 1 );
}

SI F table(const skcms_Curve* curve, F v) {
    // Clamp the input to [0,1], then scale to a table index.
    F ix = max_(F0, min_(v, F1)) * (float)(curve->table_entries - 1);

    // We'll look up (equal or adjacent) entries at lo and hi, then lerp by t between the two.
    I32 lo = cast<I32>(            ix      ),
        hi = cast<I32>(minus_1_ulp(ix+1.0f));
    F t = ix - cast<F>(lo);  // i.e. the fractional part of ix.

    // TODO: can we load l and h simultaneously?  Each entry in 'h' is either
    // the same as in 'l' or adjacent.  We have a rough idea that's it'd always be safe
    // to read adjacent entries and perhaps underflow the table by a byte or two
    // (it'd be junk, but always safe to read).  Not sure how to lerp yet.
    F l,h;
    if (curve->table_8) {
        l = F_from_U8(gather_8(curve->table_8, lo));
        h = F_from_U8(gather_8(curve->table_8, hi));
    } else {
        l = F_from_U16_BE(gather_16(curve->table_16, lo));
        h = F_from_U16_BE(gather_16(curve->table_16, hi));
    }
    return l + (h-l)*t;
}

SI void sample_clut_8(const uint8_t* grid_8, I32 ix, F* r, F* g, F* b) {
    U32 rgb = gather_24(grid_8, ix);

    *r = cast<F>((rgb >>  0) & 0xff) * (1/255.0f);
    *g = cast<F>((rgb >>  8) & 0xff) * (1/255.0f);
    *b = cast<F>((rgb >> 16) & 0xff) * (1/255.0f);
}

SI void sample_clut_8(const uint8_t* grid_8, I32 ix, F* r, F* g, F* b, F* a) {
    // TODO: don't forget to optimize gather_32().
    U32 rgba = gather_32(grid_8, ix);

    *r = cast<F>((rgba >>  0) & 0xff) * (1/255.0f);
    *g = cast<F>((rgba >>  8) & 0xff) * (1/255.0f);
    *b = cast<F>((rgba >> 16) & 0xff) * (1/255.0f);
    *a = cast<F>((rgba >> 24) & 0xff) * (1/255.0f);
}

SI void sample_clut_16(const uint8_t* grid_16, I32 ix, F* r, F* g, F* b) {
#if defined(__arm__) || defined(__loongarch_sx)
    // This is up to 2x faster on 32-bit ARM than the #else-case fast path.
    *r = F_from_U16_BE(gather_16(grid_16, 3*ix+0));
    *g = F_from_U16_BE(gather_16(grid_16, 3*ix+1));
    *b = F_from_U16_BE(gather_16(grid_16, 3*ix+2));
#else
    // This strategy is much faster for 64-bit builds, and fine for 32-bit x86 too.
    U64 rgb;
    gather_48(grid_16, ix, &rgb);
    rgb = swap_endian_16x4(rgb);

    *r = cast<F>((rgb >>  0) & 0xffff) * (1/65535.0f);
    *g = cast<F>((rgb >> 16) & 0xffff) * (1/65535.0f);
    *b = cast<F>((rgb >> 32) & 0xffff) * (1/65535.0f);
#endif
}

SI void sample_clut_16(const uint8_t* grid_16, I32 ix, F* r, F* g, F* b, F* a) {
    // TODO: gather_64()-based fast path?
    *r = F_from_U16_BE(gather_16(grid_16, 4*ix+0));
    *g = F_from_U16_BE(gather_16(grid_16, 4*ix+1));
    *b = F_from_U16_BE(gather_16(grid_16, 4*ix+2));
    *a = F_from_U16_BE(gather_16(grid_16, 4*ix+3));
}

static void clut(uint32_t input_channels, uint32_t output_channels,
                 const uint8_t grid_points[4], const uint8_t* grid_8, const uint8_t* grid_16,
                 F* r, F* g, F* b, F* a) {

    const int dim = (int)input_channels;
    assert (0 < dim && dim <= 4);
    assert (output_channels == 3 ||
            output_channels == 4);

    // For each of these arrays, think foo[2*dim], but we use foo[8] since we know dim <= 4.
    I32 index [8];  // Index contribution by dimension, first low from 0, then high from 4.
    F   weight[8];  // Weight for each contribution, again first low, then high.

    // O(dim) work first: calculate index,weight from r,g,b,a.
    const F inputs[] = { *r,*g,*b,*a };
    for (int i = dim-1, stride = 1; i >= 0; i--) {
        // x is where we logically want to sample the grid in the i-th dimension.
        F x = inputs[i] * (float)(grid_points[i] - 1);

        // But we can't index at floats.  lo and hi are the two integer grid points surrounding x.
        I32 lo = cast<I32>(            x      ),   // i.e. trunc(x) == floor(x) here.
            hi = cast<I32>(minus_1_ulp(x+1.0f));
        // Notice how we fold in the accumulated stride across previous dimensions here.
        index[i+0] = lo * stride;
        index[i+4] = hi * stride;
        stride *= grid_points[i];

        // We'll interpolate between those two integer grid points by t.
        F t = x - cast<F>(lo);  // i.e. fract(x)
        weight[i+0] = 1-t;
        weight[i+4] = t;
    }

    *r = *g = *b = F0;
    if (output_channels == 4) {
        *a = F0;
    }

    // We'll sample 2^dim == 1<<dim table entries per pixel,
    // in all combinations of low and high in each dimension.
    for (int combo = 0; combo < (1<<dim); combo++) {  // This loop can be done in any order.

        // Each of these upcoming (combo&N)*K expressions here evaluates to 0 or 4,
        // where 0 selects the low index contribution and its weight 1-t,
        // or 4 the high index contribution and its weight t.

        // Since 0<dim≤4, we can always just start off with the 0-th channel,
        // then handle the others conditionally.
        I32 ix = index [0 + (combo&1)*4];
        F    w = weight[0 + (combo&1)*4];

        switch ((dim-1)&3) {  // This lets the compiler know there are no other cases to handle.
            case 3: ix += index [3 + (combo&8)/2];
                    w  *= weight[3 + (combo&8)/2];
                    SKCMS_FALLTHROUGH;
                    // fall through

            case 2: ix += index [2 + (combo&4)*1];
                    w  *= weight[2 + (combo&4)*1];
                    SKCMS_FALLTHROUGH;
                    // fall through

            case 1: ix += index [1 + (combo&2)*2];
                    w  *= weight[1 + (combo&2)*2];
        }

        F R,G,B,A=F0;
        if (output_channels == 3) {
            if (grid_8) { sample_clut_8 (grid_8 ,ix, &R,&G,&B); }
            else        { sample_clut_16(grid_16,ix, &R,&G,&B); }
        } else {
            if (grid_8) { sample_clut_8 (grid_8 ,ix, &R,&G,&B,&A); }
            else        { sample_clut_16(grid_16,ix, &R,&G,&B,&A); }
        }
        *r += w*R;
        *g += w*G;
        *b += w*B;
        *a += w*A;
    }
}

static void clut(const skcms_A2B* a2b, F* r, F* g, F* b, F a) {
    clut(a2b->input_channels, a2b->output_channels,
         a2b->grid_points, a2b->grid_8, a2b->grid_16,
         r,g,b,&a);
}
static void clut(const skcms_B2A* b2a, F* r, F* g, F* b, F* a) {
    clut(b2a->input_channels, b2a->output_channels,
         b2a->grid_points, b2a->grid_8, b2a->grid_16,
         r,g,b,a);
}

struct NoCtx {};

struct Ctx {
    const void* fArg;
    operator NoCtx()                    { return NoCtx{}; }
    template <typename T> operator T*() { return (const T*)fArg; }
};

#define STAGE_PARAMS(MAYBE_REF) SKCMS_MAYBE_UNUSED const char* src, \
                                SKCMS_MAYBE_UNUSED char* dst,       \
                                SKCMS_MAYBE_UNUSED F MAYBE_REF r,   \
                                SKCMS_MAYBE_UNUSED F MAYBE_REF g,   \
                                SKCMS_MAYBE_UNUSED F MAYBE_REF b,   \
                                SKCMS_MAYBE_UNUSED F MAYBE_REF a,   \
                                SKCMS_MAYBE_UNUSED int i

#if SKCMS_HAS_MUSTTAIL

    // Stages take a stage list, and each stage is responsible for tail-calling the next one.
    //
    // Unfortunately, we can't declare a StageFn as a function pointer which takes a pointer to
    // another StageFn; declaring this leads to a circular dependency. To avoid this, StageFn is
    // wrapped in a single-element `struct StageList` which we are able to forward-declare.
    struct StageList;
    using StageFn = void (*)(StageList stages, const void** ctx, STAGE_PARAMS());
    struct StageList {
        const StageFn* fn;
    };

    #define DECLARE_STAGE(name, arg, CALL_NEXT)                                 \
        SI void Exec_##name##_k(arg, STAGE_PARAMS(&));                          \
                                                                                \
        SI void Exec_##name(StageList list, const void** ctx, STAGE_PARAMS()) { \
            Exec_##name##_k(Ctx{*ctx}, src, dst, r, g, b, a, i);                \
            ++list.fn; ++ctx;                                                   \
            CALL_NEXT;                                                          \
        }                                                                       \
                                                                                \
        SI void Exec_##name##_k(arg, STAGE_PARAMS(&))

    #define STAGE(name, arg)                                                                \
        DECLARE_STAGE(name, arg, [[clang::musttail]] return (*list.fn)(list, ctx, src, dst, \
                                                                       r, g, b, a, i))

    #define FINAL_STAGE(name, arg) \
        DECLARE_STAGE(name, arg, /* Stop executing stages and return to the caller. */)

#else

    #define DECLARE_STAGE(name, arg)                            \
        SI void Exec_##name##_k(arg, STAGE_PARAMS(&));          \
                                                                \
        SI void Exec_##name(const void* ctx, STAGE_PARAMS(&)) { \
            Exec_##name##_k(Ctx{ctx}, src, dst, r, g, b, a, i); \
        }                                                       \
                                                                \
        SI void Exec_##name##_k(arg, STAGE_PARAMS(&))

    #define STAGE(name, arg)       DECLARE_STAGE(name, arg)
    #define FINAL_STAGE(name, arg) DECLARE_STAGE(name, arg)

#endif

STAGE(load_a8, NoCtx) {
    a = F_from_U8(load<U8>(src + 1*i));
}

STAGE(load_g8, NoCtx) {
    r = g = b = F_from_U8(load<U8>(src + 1*i));
}

STAGE(load_ga88, NoCtx) {
    U16 u16 = load<U16>(src + 2 * i);
    r = g = b = cast<F>((u16 >> 0) & 0xff) * (1 / 255.0f);
            a = cast<F>((u16 >> 8) & 0xff) * (1 / 255.0f);
}

STAGE(load_4444, NoCtx) {
    U16 abgr = load<U16>(src + 2*i);

    r = cast<F>((abgr >> 12) & 0xf) * (1/15.0f);
    g = cast<F>((abgr >>  8) & 0xf) * (1/15.0f);
    b = cast<F>((abgr >>  4) & 0xf) * (1/15.0f);
    a = cast<F>((abgr >>  0) & 0xf) * (1/15.0f);
}

STAGE(load_565, NoCtx) {
    U16 rgb = load<U16>(src + 2*i);

    r = cast<F>(rgb & (uint16_t)(31<< 0)) * (1.0f / (31<< 0));
    g = cast<F>(rgb & (uint16_t)(63<< 5)) * (1.0f / (63<< 5));
    b = cast<F>(rgb & (uint16_t)(31<<11)) * (1.0f / (31<<11));
}

STAGE(load_888, NoCtx) {
    const uint8_t* rgb = (const uint8_t*)(src + 3*i);
#if defined(USING_NEON)
    // There's no uint8x4x3_t or vld3 load for it, so we'll load each rgb pixel one at
    // a time.  Since we're doing that, we might as well load them into 16-bit lanes.
    // (We'd even load into 32-bit lanes, but that's not possible on ARMv7.)
    uint8x8x3_t v = {{ vdup_n_u8(0), vdup_n_u8(0), vdup_n_u8(0) }};
    v = vld3_lane_u8(rgb+0, v, 0);
    v = vld3_lane_u8(rgb+3, v, 2);
    v = vld3_lane_u8(rgb+6, v, 4);
    v = vld3_lane_u8(rgb+9, v, 6);

    // Now if we squint, those 3 uint8x8_t we constructed are really U16s, easy to
    // convert to F.  (Again, U32 would be even better here if drop ARMv7 or split
    // ARMv7 and ARMv8 impls.)
    r = cast<F>((U16)v.val[0]) * (1/255.0f);
    g = cast<F>((U16)v.val[1]) * (1/255.0f);
    b = cast<F>((U16)v.val[2]) * (1/255.0f);
#else
    r = cast<F>(load_3<U32>(rgb+0) ) * (1/255.0f);
    g = cast<F>(load_3<U32>(rgb+1) ) * (1/255.0f);
    b = cast<F>(load_3<U32>(rgb+2) ) * (1/255.0f);
#endif
}

STAGE(load_8888, NoCtx) {
    U32 rgba = load<U32>(src + 4*i);

    r = cast<F>((rgba >>  0) & 0xff) * (1/255.0f);
    g = cast<F>((rgba >>  8) & 0xff) * (1/255.0f);
    b = cast<F>((rgba >> 16) & 0xff) * (1/255.0f);
    a = cast<F>((rgba >> 24) & 0xff) * (1/255.0f);
}

STAGE(load_1010102, NoCtx) {
    U32 rgba = load<U32>(src + 4*i);

    r = cast<F>((rgba >>  0) & 0x3ff) * (1/1023.0f);
    g = cast<F>((rgba >> 10) & 0x3ff) * (1/1023.0f);
    b = cast<F>((rgba >> 20) & 0x3ff) * (1/1023.0f);
    a = cast<F>((rgba >> 30) & 0x3  ) * (1/   3.0f);
}

STAGE(load_101010x_XR, NoCtx) {
    static constexpr float min = -0.752941f;
    static constexpr float max = 1.25098f;
    static constexpr float range = max - min;
    U32 rgba = load<U32>(src + 4*i);
    r = cast<F>((rgba >>  0) & 0x3ff) * (1/1023.0f) * range + min;
    g = cast<F>((rgba >> 10) & 0x3ff) * (1/1023.0f) * range + min;
    b = cast<F>((rgba >> 20) & 0x3ff) * (1/1023.0f) * range + min;
}

STAGE(load_10101010_XR, NoCtx) {
    static constexpr float min = -0.752941f;
    static constexpr float max = 1.25098f;
    static constexpr float range = max - min;
    U64 rgba = load<U64>(src + 8 * i);
    r = cast<F>((rgba >>  (0+6)) & 0x3ff) * (1/1023.0f) * range + min;
    g = cast<F>((rgba >> (16+6)) & 0x3ff) * (1/1023.0f) * range + min;
    b = cast<F>((rgba >> (32+6)) & 0x3ff) * (1/1023.0f) * range + min;
    a = cast<F>((rgba >> (48+6)) & 0x3ff) * (1/1023.0f) * range + min;
}

STAGE(load_161616LE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 6*i);
    assert( (ptr & 1) == 0 );                   // src must be 2-byte aligned for this
    const uint16_t* rgb = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x3_t v = vld3_u16(rgb);
    r = cast<F>((U16)v.val[0]) * (1/65535.0f);
    g = cast<F>((U16)v.val[1]) * (1/65535.0f);
    b = cast<F>((U16)v.val[2]) * (1/65535.0f);
#else
    r = cast<F>(load_3<U32>(rgb+0)) * (1/65535.0f);
    g = cast<F>(load_3<U32>(rgb+1)) * (1/65535.0f);
    b = cast<F>(load_3<U32>(rgb+2)) * (1/65535.0f);
#endif
}

STAGE(load_16161616LE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 8*i);
    assert( (ptr & 1) == 0 );                    // src must be 2-byte aligned for this
    const uint16_t* rgba = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x4_t v = vld4_u16(rgba);
    r = cast<F>((U16)v.val[0]) * (1/65535.0f);
    g = cast<F>((U16)v.val[1]) * (1/65535.0f);
    b = cast<F>((U16)v.val[2]) * (1/65535.0f);
    a = cast<F>((U16)v.val[3]) * (1/65535.0f);
#else
    U64 px = load<U64>(rgba);

    r = cast<F>((px >>  0) & 0xffff) * (1/65535.0f);
    g = cast<F>((px >> 16) & 0xffff) * (1/65535.0f);
    b = cast<F>((px >> 32) & 0xffff) * (1/65535.0f);
    a = cast<F>((px >> 48) & 0xffff) * (1/65535.0f);
#endif
}

STAGE(load_161616BE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 6*i);
    assert( (ptr & 1) == 0 );                   // src must be 2-byte aligned for this
    const uint16_t* rgb = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x3_t v = vld3_u16(rgb);
    r = cast<F>(swap_endian_16((U16)v.val[0])) * (1/65535.0f);
    g = cast<F>(swap_endian_16((U16)v.val[1])) * (1/65535.0f);
    b = cast<F>(swap_endian_16((U16)v.val[2])) * (1/65535.0f);
#else
    U32 R = load_3<U32>(rgb+0),
        G = load_3<U32>(rgb+1),
        B = load_3<U32>(rgb+2);
    // R,G,B are big-endian 16-bit, so byte swap them before converting to float.
    r = cast<F>((R & 0x00ff)<<8 | (R & 0xff00)>>8) * (1/65535.0f);
    g = cast<F>((G & 0x00ff)<<8 | (G & 0xff00)>>8) * (1/65535.0f);
    b = cast<F>((B & 0x00ff)<<8 | (B & 0xff00)>>8) * (1/65535.0f);
#endif
}

STAGE(load_16161616BE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 8*i);
    assert( (ptr & 1) == 0 );                    // src must be 2-byte aligned for this
    const uint16_t* rgba = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x4_t v = vld4_u16(rgba);
    r = cast<F>(swap_endian_16((U16)v.val[0])) * (1/65535.0f);
    g = cast<F>(swap_endian_16((U16)v.val[1])) * (1/65535.0f);
    b = cast<F>(swap_endian_16((U16)v.val[2])) * (1/65535.0f);
    a = cast<F>(swap_endian_16((U16)v.val[3])) * (1/65535.0f);
#else
    U64 px = swap_endian_16x4(load<U64>(rgba));

    r = cast<F>((px >>  0) & 0xffff) * (1/65535.0f);
    g = cast<F>((px >> 16) & 0xffff) * (1/65535.0f);
    b = cast<F>((px >> 32) & 0xffff) * (1/65535.0f);
    a = cast<F>((px >> 48) & 0xffff) * (1/65535.0f);
#endif
}

STAGE(load_hhh, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 6*i);
    assert( (ptr & 1) == 0 );                   // src must be 2-byte aligned for this
    const uint16_t* rgb = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x3_t v = vld3_u16(rgb);
    U16 R = (U16)v.val[0],
        G = (U16)v.val[1],
        B = (U16)v.val[2];
#else
    U16 R = load_3<U16>(rgb+0),
        G = load_3<U16>(rgb+1),
        B = load_3<U16>(rgb+2);
#endif
    r = F_from_Half(R);
    g = F_from_Half(G);
    b = F_from_Half(B);
}

STAGE(load_hhhh, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 8*i);
    assert( (ptr & 1) == 0 );                    // src must be 2-byte aligned for this
    const uint16_t* rgba = (const uint16_t*)ptr; // cast to const uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x4_t v = vld4_u16(rgba);
    U16 R = (U16)v.val[0],
        G = (U16)v.val[1],
        B = (U16)v.val[2],
        A = (U16)v.val[3];
#else
    U64 px = load<U64>(rgba);
    U16 R = cast<U16>((px >>  0) & 0xffff),
        G = cast<U16>((px >> 16) & 0xffff),
        B = cast<U16>((px >> 32) & 0xffff),
        A = cast<U16>((px >> 48) & 0xffff);
#endif
    r = F_from_Half(R);
    g = F_from_Half(G);
    b = F_from_Half(B);
    a = F_from_Half(A);
}

STAGE(load_fff, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 12*i);
    assert( (ptr & 3) == 0 );                   // src must be 4-byte aligned for this
    const float* rgb = (const float*)ptr;       // cast to const float* to be safe.
#if defined(USING_NEON)
    float32x4x3_t v = vld3q_f32(rgb);
    r = (F)v.val[0];
    g = (F)v.val[1];
    b = (F)v.val[2];
#else
    r = load_3<F>(rgb+0);
    g = load_3<F>(rgb+1);
    b = load_3<F>(rgb+2);
#endif
}

STAGE(load_ffff, NoCtx) {
    uintptr_t ptr = (uintptr_t)(src + 16*i);
    assert( (ptr & 3) == 0 );                   // src must be 4-byte aligned for this
    const float* rgba = (const float*)ptr;      // cast to const float* to be safe.
#if defined(USING_NEON)
    float32x4x4_t v = vld4q_f32(rgba);
    r = (F)v.val[0];
    g = (F)v.val[1];
    b = (F)v.val[2];
    a = (F)v.val[3];
#else
    r = load_4<F>(rgba+0);
    g = load_4<F>(rgba+1);
    b = load_4<F>(rgba+2);
    a = load_4<F>(rgba+3);
#endif
}

STAGE(swap_rb, NoCtx) {
    F t = r;
    r = b;
    b = t;
}

STAGE(clamp, NoCtx) {
    r = max_(F0, min_(r, F1));
    g = max_(F0, min_(g, F1));
    b = max_(F0, min_(b, F1));
    a = max_(F0, min_(a, F1));
}

STAGE(invert, NoCtx) {
    r = F1 - r;
    g = F1 - g;
    b = F1 - b;
    a = F1 - a;
}

STAGE(force_opaque, NoCtx) {
    a = F1;
}

STAGE(premul, NoCtx) {
    r *= a;
    g *= a;
    b *= a;
}

STAGE(unpremul, NoCtx) {
    F scale = if_then_else(F1 / a < INFINITY_, F1 / a, F0);
    r *= scale;
    g *= scale;
    b *= scale;
}

STAGE(matrix_3x3, const skcms_Matrix3x3* matrix) {
    const float* m = &matrix->vals[0][0];

    F R = m[0]*r + m[1]*g + m[2]*b,
      G = m[3]*r + m[4]*g + m[5]*b,
      B = m[6]*r + m[7]*g + m[8]*b;

    r = R;
    g = G;
    b = B;
}

STAGE(matrix_3x4, const skcms_Matrix3x4* matrix) {
    const float* m = &matrix->vals[0][0];

    F R = m[0]*r + m[1]*g + m[ 2]*b + m[ 3],
      G = m[4]*r + m[5]*g + m[ 6]*b + m[ 7],
      B = m[8]*r + m[9]*g + m[10]*b + m[11];

    r = R;
    g = G;
    b = B;
}

STAGE(lab_to_xyz, NoCtx) {
    // The L*a*b values are in r,g,b, but normalized to [0,1].  Reconstruct them:
    F L = r * 100.0f,
      A = g * 255.0f - 128.0f,
      B = b * 255.0f - 128.0f;

    // Convert to CIE XYZ.
    F Y = (L + 16.0f) * (1/116.0f),
      X = Y + A*(1/500.0f),
      Z = Y - B*(1/200.0f);

    X = if_then_else(X*X*X > 0.008856f, X*X*X, (X - (16/116.0f)) * (1/7.787f));
    Y = if_then_else(Y*Y*Y > 0.008856f, Y*Y*Y, (Y - (16/116.0f)) * (1/7.787f));
    Z = if_then_else(Z*Z*Z > 0.008856f, Z*Z*Z, (Z - (16/116.0f)) * (1/7.787f));

    // Adjust to XYZD50 illuminant, and stuff back into r,g,b for the next op.
    r = X * 0.9642f;
    g = Y          ;
    b = Z * 0.8249f;
}

// As above, in reverse.
STAGE(xyz_to_lab, NoCtx) {
    F X = r * (1/0.9642f),
      Y = g,
      Z = b * (1/0.8249f);

    X = if_then_else(X > 0.008856f, approx_pow(X, 1/3.0f), X*7.787f + (16/116.0f));
    Y = if_then_else(Y > 0.008856f, approx_pow(Y, 1/3.0f), Y*7.787f + (16/116.0f));
    Z = if_then_else(Z > 0.008856f, approx_pow(Z, 1/3.0f), Z*7.787f + (16/116.0f));

    F L = Y*116.0f - 16.0f,
      A = (X-Y)*500.0f,
      B = (Y-Z)*200.0f;

    r = L * (1/100.f);
    g = (A + 128.0f) * (1/255.0f);
    b = (B + 128.0f) * (1/255.0f);
}

STAGE(gamma_r, const skcms_TransferFunction* tf) { r = apply_gamma(tf, r); }
STAGE(gamma_g, const skcms_TransferFunction* tf) { g = apply_gamma(tf, g); }
STAGE(gamma_b, const skcms_TransferFunction* tf) { b = apply_gamma(tf, b); }
STAGE(gamma_a, const skcms_TransferFunction* tf) { a = apply_gamma(tf, a); }

STAGE(gamma_rgb, const skcms_TransferFunction* tf) {
    r = apply_gamma(tf, r);
    g = apply_gamma(tf, g);
    b = apply_gamma(tf, b);
}

STAGE(tf_r, const skcms_TransferFunction* tf) { r = apply_tf(tf, r); }
STAGE(tf_g, const skcms_TransferFunction* tf) { g = apply_tf(tf, g); }
STAGE(tf_b, const skcms_TransferFunction* tf) { b = apply_tf(tf, b); }
STAGE(tf_a, const skcms_TransferFunction* tf) { a = apply_tf(tf, a); }

STAGE(tf_rgb, const skcms_TransferFunction* tf) {
    r = apply_tf(tf, r);
    g = apply_tf(tf, g);
    b = apply_tf(tf, b);
}

STAGE(pq_r, const skcms_TransferFunction* tf) { r = apply_pq(tf, r); }
STAGE(pq_g, const skcms_TransferFunction* tf) { g = apply_pq(tf, g); }
STAGE(pq_b, const skcms_TransferFunction* tf) { b = apply_pq(tf, b); }
STAGE(pq_a, const skcms_TransferFunction* tf) { a = apply_pq(tf, a); }

STAGE(pq_rgb, const skcms_TransferFunction* tf) {
    r = apply_pq(tf, r);
    g = apply_pq(tf, g);
    b = apply_pq(tf, b);
}

STAGE(hlg_r, const skcms_TransferFunction* tf) { r = apply_hlg(tf, r); }
STAGE(hlg_g, const skcms_TransferFunction* tf) { g = apply_hlg(tf, g); }
STAGE(hlg_b, const skcms_TransferFunction* tf) { b = apply_hlg(tf, b); }
STAGE(hlg_a, const skcms_TransferFunction* tf) { a = apply_hlg(tf, a); }

STAGE(hlg_rgb, const skcms_TransferFunction* tf) {
    r = apply_hlg(tf, r);
    g = apply_hlg(tf, g);
    b = apply_hlg(tf, b);
}

STAGE(hlginv_r, const skcms_TransferFunction* tf) { r = apply_hlginv(tf, r); }
STAGE(hlginv_g, const skcms_TransferFunction* tf) { g = apply_hlginv(tf, g); }
STAGE(hlginv_b, const skcms_TransferFunction* tf) { b = apply_hlginv(tf, b); }
STAGE(hlginv_a, const skcms_TransferFunction* tf) { a = apply_hlginv(tf, a); }

STAGE(hlginv_rgb, const skcms_TransferFunction* tf) {
    r = apply_hlginv(tf, r);
    g = apply_hlginv(tf, g);
    b = apply_hlginv(tf, b);
}

STAGE(table_r, const skcms_Curve* curve) { r = table(curve, r); }
STAGE(table_g, const skcms_Curve* curve) { g = table(curve, g); }
STAGE(table_b, const skcms_Curve* curve) { b = table(curve, b); }
STAGE(table_a, const skcms_Curve* curve) { a = table(curve, a); }

STAGE(clut_A2B, const skcms_A2B* a2b) {
    clut(a2b, &r,&g,&b,a);

    if (a2b->input_channels == 4) {
        // CMYK is opaque.
        a = F1;
    }
}

STAGE(clut_B2A, const skcms_B2A* b2a) {
    clut(b2a, &r,&g,&b,&a);
}

// From here on down, the store_ ops are all "final stages," terminating processing of this group.

FINAL_STAGE(store_a8, NoCtx) {
    store(dst + 1*i, cast<U8>(to_fixed(a * 255)));
}

FINAL_STAGE(store_g8, NoCtx) {
    // g should be holding luminance (Y) (r,g,b ~~~> X,Y,Z)
    store(dst + 1*i, cast<U8>(to_fixed(g * 255)));
}

FINAL_STAGE(store_ga88, NoCtx) {
    // g should be holding luminance (Y) (r,g,b ~~~> X,Y,Z)
    store<U16>(dst + 2*i, cast<U16>(to_fixed(g * 255) << 0 )
                        | cast<U16>(to_fixed(a * 255) << 8 ));
}

FINAL_STAGE(store_4444, NoCtx) {
    store<U16>(dst + 2*i, cast<U16>(to_fixed(r * 15) << 12)
                        | cast<U16>(to_fixed(g * 15) <<  8)
                        | cast<U16>(to_fixed(b * 15) <<  4)
                        | cast<U16>(to_fixed(a * 15) <<  0));
}

FINAL_STAGE(store_565, NoCtx) {
    store<U16>(dst + 2*i, cast<U16>(to_fixed(r * 31) <<  0 )
                        | cast<U16>(to_fixed(g * 63) <<  5 )
                        | cast<U16>(to_fixed(b * 31) << 11 ));
}

FINAL_STAGE(store_888, NoCtx) {
    uint8_t* rgb = (uint8_t*)dst + 3*i;
#if defined(USING_NEON)
    // Same deal as load_888 but in reverse... we'll store using uint8x8x3_t, but
    // get there via U16 to save some instructions converting to float.  And just
    // like load_888, we'd prefer to go via U32 but for ARMv7 support.
    U16 R = cast<U16>(to_fixed(r * 255)),
        G = cast<U16>(to_fixed(g * 255)),
        B = cast<U16>(to_fixed(b * 255));

    uint8x8x3_t v = {{ (uint8x8_t)R, (uint8x8_t)G, (uint8x8_t)B }};
    vst3_lane_u8(rgb+0, v, 0);
    vst3_lane_u8(rgb+3, v, 2);
    vst3_lane_u8(rgb+6, v, 4);
    vst3_lane_u8(rgb+9, v, 6);
#else
    store_3(rgb+0, cast<U8>(to_fixed(r * 255)) );
    store_3(rgb+1, cast<U8>(to_fixed(g * 255)) );
    store_3(rgb+2, cast<U8>(to_fixed(b * 255)) );
#endif
}

FINAL_STAGE(store_8888, NoCtx) {
    store(dst + 4*i, cast<U32>(to_fixed(r * 255)) <<  0
                   | cast<U32>(to_fixed(g * 255)) <<  8
                   | cast<U32>(to_fixed(b * 255)) << 16
                   | cast<U32>(to_fixed(a * 255)) << 24);
}

FINAL_STAGE(store_101010x_XR, NoCtx) {
    static constexpr float min = -0.752941f;
    static constexpr float max = 1.25098f;
    static constexpr float range = max - min;
    store(dst + 4*i, cast<U32>(to_fixed(((r - min) / range) * 1023)) <<  0
                   | cast<U32>(to_fixed(((g - min) / range) * 1023)) << 10
                   | cast<U32>(to_fixed(((b - min) / range) * 1023)) << 20);
}

FINAL_STAGE(store_1010102, NoCtx) {
    store(dst + 4*i, cast<U32>(to_fixed(r * 1023)) <<  0
                   | cast<U32>(to_fixed(g * 1023)) << 10
                   | cast<U32>(to_fixed(b * 1023)) << 20
                   | cast<U32>(to_fixed(a *    3)) << 30);
}

FINAL_STAGE(store_161616LE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 6*i);
    assert( (ptr & 1) == 0 );                // The dst pointer must be 2-byte aligned
    uint16_t* rgb = (uint16_t*)ptr;          // for this cast to uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x3_t v = {{
        (uint16x4_t)U16_from_F(r),
        (uint16x4_t)U16_from_F(g),
        (uint16x4_t)U16_from_F(b),
    }};
    vst3_u16(rgb, v);
#else
    store_3(rgb+0, U16_from_F(r));
    store_3(rgb+1, U16_from_F(g));
    store_3(rgb+2, U16_from_F(b));
#endif

}

FINAL_STAGE(store_16161616LE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 8*i);
    assert( (ptr & 1) == 0 );               // The dst pointer must be 2-byte aligned
    uint16_t* rgba = (uint16_t*)ptr;        // for this cast to uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x4_t v = {{
        (uint16x4_t)U16_from_F(r),
        (uint16x4_t)U16_from_F(g),
        (uint16x4_t)U16_from_F(b),
        (uint16x4_t)U16_from_F(a),
    }};
    vst4_u16(rgba, v);
#else
    U64 px = cast<U64>(to_fixed(r * 65535)) <<  0
           | cast<U64>(to_fixed(g * 65535)) << 16
           | cast<U64>(to_fixed(b * 65535)) << 32
           | cast<U64>(to_fixed(a * 65535)) << 48;
    store(rgba, px);
#endif
}

FINAL_STAGE(store_161616BE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 6*i);
    assert( (ptr & 1) == 0 );                // The dst pointer must be 2-byte aligned
    uint16_t* rgb = (uint16_t*)ptr;          // for this cast to uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x3_t v = {{
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(r))),
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(g))),
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(b))),
    }};
    vst3_u16(rgb, v);
#else
    U32 R = to_fixed(r * 65535),
        G = to_fixed(g * 65535),
        B = to_fixed(b * 65535);
    store_3(rgb+0, cast<U16>((R & 0x00ff) << 8 | (R & 0xff00) >> 8) );
    store_3(rgb+1, cast<U16>((G & 0x00ff) << 8 | (G & 0xff00) >> 8) );
    store_3(rgb+2, cast<U16>((B & 0x00ff) << 8 | (B & 0xff00) >> 8) );
#endif

}

FINAL_STAGE(store_16161616BE, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 8*i);
    assert( (ptr & 1) == 0 );               // The dst pointer must be 2-byte aligned
    uint16_t* rgba = (uint16_t*)ptr;        // for this cast to uint16_t* to be safe.
#if defined(USING_NEON)
    uint16x4x4_t v = {{
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(r))),
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(g))),
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(b))),
        (uint16x4_t)swap_endian_16(cast<U16>(U16_from_F(a))),
    }};
    vst4_u16(rgba, v);
#else
    U64 px = cast<U64>(to_fixed(r * 65535)) <<  0
           | cast<U64>(to_fixed(g * 65535)) << 16
           | cast<U64>(to_fixed(b * 65535)) << 32
           | cast<U64>(to_fixed(a * 65535)) << 48;
    store(rgba, swap_endian_16x4(px));
#endif
}

FINAL_STAGE(store_hhh, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 6*i);
    assert( (ptr & 1) == 0 );                // The dst pointer must be 2-byte aligned
    uint16_t* rgb = (uint16_t*)ptr;          // for this cast to uint16_t* to be safe.

    U16 R = Half_from_F(r),
        G = Half_from_F(g),
        B = Half_from_F(b);
#if defined(USING_NEON)
    uint16x4x3_t v = {{
        (uint16x4_t)R,
        (uint16x4_t)G,
        (uint16x4_t)B,
    }};
    vst3_u16(rgb, v);
#else
    store_3(rgb+0, R);
    store_3(rgb+1, G);
    store_3(rgb+2, B);
#endif
}

FINAL_STAGE(store_hhhh, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 8*i);
    assert( (ptr & 1) == 0 );                // The dst pointer must be 2-byte aligned
    uint16_t* rgba = (uint16_t*)ptr;         // for this cast to uint16_t* to be safe.

    U16 R = Half_from_F(r),
        G = Half_from_F(g),
        B = Half_from_F(b),
        A = Half_from_F(a);
#if defined(USING_NEON)
    uint16x4x4_t v = {{
        (uint16x4_t)R,
        (uint16x4_t)G,
        (uint16x4_t)B,
        (uint16x4_t)A,
    }};
    vst4_u16(rgba, v);
#else
    store(rgba, cast<U64>(R) <<  0
              | cast<U64>(G) << 16
              | cast<U64>(B) << 32
              | cast<U64>(A) << 48);
#endif
}

FINAL_STAGE(store_fff, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 12*i);
    assert( (ptr & 3) == 0 );                // The dst pointer must be 4-byte aligned
    float* rgb = (float*)ptr;                // for this cast to float* to be safe.
#if defined(USING_NEON)
    float32x4x3_t v = {{
        (float32x4_t)r,
        (float32x4_t)g,
        (float32x4_t)b,
    }};
    vst3q_f32(rgb, v);
#else
    store_3(rgb+0, r);
    store_3(rgb+1, g);
    store_3(rgb+2, b);
#endif
}

FINAL_STAGE(store_ffff, NoCtx) {
    uintptr_t ptr = (uintptr_t)(dst + 16*i);
    assert( (ptr & 3) == 0 );                // The dst pointer must be 4-byte aligned
    float* rgba = (float*)ptr;               // for this cast to float* to be safe.
#if defined(USING_NEON)
    float32x4x4_t v = {{
        (float32x4_t)r,
        (float32x4_t)g,
        (float32x4_t)b,
        (float32x4_t)a,
    }};
    vst4q_f32(rgba, v);
#else
    store_4(rgba+0, r);
    store_4(rgba+1, g);
    store_4(rgba+2, b);
    store_4(rgba+3, a);
#endif
}

#if SKCMS_HAS_MUSTTAIL

    SI void exec_stages(StageFn* stages, const void** contexts, const char* src, char* dst, int i) {
        (*stages)({stages}, contexts, src, dst, F0, F0, F0, F1, i);
    }

#else

    static void exec_stages(const Op* ops, const void** contexts,
                            const char* src, char* dst, int i) {
        F r = F0, g = F0, b = F0, a = F1;
        while (true) {
            switch (*ops++) {
#define M(name) case Op::name: Exec_##name(*contexts++, src, dst, r, g, b, a, i); break;
                SKCMS_WORK_OPS(M)
#undef M
#define M(name) case Op::name: Exec_##name(*contexts++, src, dst, r, g, b, a, i); return;
                SKCMS_STORE_OPS(M)
#undef M
            }
        }
    }

#endif

// NOLINTNEXTLINE(misc-definitions-in-headers)
void run_program(const Op* program, const void** contexts, SKCMS_MAYBE_UNUSED ptrdiff_t programSize,
                 const char* src, char* dst, int n,
                 const size_t src_bpp, const size_t dst_bpp) {
#if SKCMS_HAS_MUSTTAIL
    // Convert the program into an array of tailcall stages.
    StageFn stages[32];
    assert(programSize <= ARRAY_COUNT(stages));

    static constexpr StageFn kStageFns[] = {
#define M(name) &Exec_##name,
        SKCMS_WORK_OPS(M)
        SKCMS_STORE_OPS(M)
#undef M
    };

    for (ptrdiff_t index = 0; index < programSize; ++index) {
        stages[index] = kStageFns[(int)program[index]];
    }
#else
    // Use the op array as-is.
    const Op* stages = program;
#endif

    int i = 0;
    while (n >= N) {
        exec_stages(stages, contexts, src, dst, i);
        i += N;
        n -= N;
    }
    if (n > 0) {
        char tmp[4*4*N] = {0};

        memcpy(tmp, (const char*)src + (size_t)i*src_bpp, (size_t)n*src_bpp);
        exec_stages(stages, contexts, tmp, tmp, 0);
        memcpy((char*)dst + (size_t)i*dst_bpp, tmp, (size_t)n*dst_bpp);
    }
}
