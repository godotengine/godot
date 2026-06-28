#pragma once

#if !defined(__aarch64__)
#error "avx2neon is only supported for AARCH64"
#endif

#include "sse2neon.h"

#define AVX2NEON_ABI static inline  __attribute__((always_inline))


struct __m256 {
    __m128 lo,hi;
    __m256() {}
};




struct __m256i {
    __m128i lo,hi;
    explicit __m256i(const __m256 a) : lo(__m128i(a.lo)),hi(__m128i(a.hi)) {}
    operator __m256() const {__m256 res; res.lo = __m128(lo);res.hi = __m128(hi); return res;}
    __m256i() {}
};




struct __m256d {
    float64x2_t lo,hi;
    __m256d() {}
    __m256d(const __m256& a) : lo(float64x2_t(a.lo)),hi(float64x2_t(a.hi)) {}
    __m256d(const __m256i& a) : lo(float64x2_t(a.lo)),hi(float64x2_t(a.hi)) {}
};

#define UNARY_AVX_OP(type,func,basic_func) AVX2NEON_ABI type func(const type& a) {type res;res.lo=basic_func(a.lo);res.hi=basic_func(a.hi);return res;}


#define BINARY_AVX_OP(type,func,basic_func) AVX2NEON_ABI type func(const type& a,const type& b) {type res;res.lo=basic_func(a.lo,b.lo);res.hi=basic_func(a.hi,b.hi);return res;}
#define BINARY_AVX_OP_CAST(type,func,basic_func,bdst,bsrc) AVX2NEON_ABI type func(const type& a,const type& b) {type res;res.lo=bdst(basic_func(bsrc(a.lo),bsrc(b.lo)));res.hi=bdst(basic_func(bsrc(a.hi),bsrc(b.hi)));return res;}

#define TERNARY_AVX_OP(type,func,basic_func) AVX2NEON_ABI type func(const type& a,const type& b,const type& c) {type res;res.lo=basic_func(a.lo,b.lo,c.lo);res.hi=basic_func(a.hi,b.hi,c.hi);return res;}


#define CAST_SIMD_TYPE(to,name,from,basic_dst) AVX2NEON_ABI to name(const from& a) { to res; res.lo = basic_dst(a.lo); res.hi=basic_dst(a.hi); return res;}



#define _mm_stream_load_si128 _mm_load_si128
#define _mm256_stream_load_si256 _mm256_load_si256


AVX2NEON_ABI
__m128i _mm_blend_epi32 (__m128i a, __m128i b, const int imm8)
{
    __m128 af = _mm_castsi128_ps(a);
    __m128 bf = _mm_castsi128_ps(b);
    __m128 blendf = _mm_blend_ps(af, bf, imm8);
    return _mm_castps_si128(blendf);
}

AVX2NEON_ABI
int _mm_movemask_popcnt(__m128 a)
{
    return __builtin_popcount(_mm_movemask_ps(a));
}

AVX2NEON_ABI
__m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)
{
    float32x4_t res;
    uint32x4_t mask_u32 = vreinterpretq_u32_m128i(mask);
    for (int i=0;i<4;i++) {
        if (mask_u32[i] & 0x80000000) res[i] = mem_addr[i]; else res[i] = 0;
    }
    return vreinterpretq_m128_f32(res);
}

AVX2NEON_ABI
void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)
{
    float32x4_t a_f32 = vreinterpretq_f32_m128(a);
    uint32x4_t mask_u32 = vreinterpretq_u32_m128i(mask);
    for (int i=0;i<4;i++) {
        if (mask_u32[i] & 0x80000000) mem_addr[i] = a_f32[i];
    }
}

AVX2NEON_ABI
void _mm_maskstore_epi32 (int * mem_addr, __m128i mask, __m128i a)
{
    uint32x4_t mask_u32 = vreinterpretq_u32_m128i(mask);
    int32x4_t a_s32 = vreinterpretq_s32_m128i(a);
    for (int i=0;i<4;i++) {
        if (mask_u32[i] & 0x80000000) mem_addr[i] = a_s32[i];
    }
}


#define _mm_fmadd_ss _mm_fmadd_ps
#define _mm_fmsub_ss _mm_fmsub_ps
#define _mm_fnmsub_ss _mm_fnmsub_ps
#define _mm_fnmadd_ss _mm_fnmadd_ps

template<int code>
AVX2NEON_ABI float32x4_t dpps_neon(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    v = 0;
    v += (code & 0x10) ? a[0]*b[0] : 0;
    v += (code & 0x20) ? a[1]*b[1] : 0;
    v += (code & 0x40) ? a[2]*b[2] : 0;
    v += (code & 0x80) ? a[3]*b[3] : 0;
    float32x4_t res;
    res[0] = (code & 0x1) ? v : 0;
    res[1] = (code & 0x2) ? v : 0;
    res[2] = (code & 0x4) ? v : 0;
    res[3] = (code & 0x8) ? v : 0;
    return res;
}

template<>
inline float32x4_t dpps_neon<0x7f>(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    float32x4_t m = _mm_mul_ps(a,b);
    m[3] = 0;
    v = vaddvq_f32(m);
    return _mm_set1_ps(v);
}

template<>
inline float32x4_t dpps_neon<0xff>(const float32x4_t& a,const float32x4_t& b)
{
    float v;
    float32x4_t m = _mm_mul_ps(a,b);
    v = vaddvq_f32(m);
    return _mm_set1_ps(v);
}

#define _mm_dp_ps(a,b,c) dpps_neon<c>((a),(b))


AVX2NEON_ABI
__m128 _mm_permutevar_ps (__m128 a, __m128i b)
{
    uint32x4_t b_u32 = vreinterpretq_u32_m128i(b);
    float32x4_t x;
    for (int i=0;i<4;i++)
    {
        x[i] = a[b_u32[i]];
    }
    return vreinterpretq_m128_f32(x);
}

AVX2NEON_ABI
__m256i _mm256_setzero_si256()
{
    __m256i res;
    res.lo = res.hi = vdupq_n_s32(0);
    return res;
}

AVX2NEON_ABI
__m256 _mm256_setzero_ps()
{
    __m256 res;
    res.lo = res.hi = vdupq_n_f32(0.0f);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_undefined_si256()
{
    return _mm256_setzero_si256();
}

AVX2NEON_ABI
__m256 _mm256_undefined_ps()
{
    return _mm256_setzero_ps();
}

CAST_SIMD_TYPE(__m256d, _mm256_castps_pd,    __m256,  float64x2_t)
CAST_SIMD_TYPE(__m256i, _mm256_castps_si256, __m256,  __m128i)
CAST_SIMD_TYPE(__m256,  _mm256_castsi256_ps, __m256i, __m128)
CAST_SIMD_TYPE(__m256,  _mm256_castpd_ps ,   __m256d, __m128)
CAST_SIMD_TYPE(__m256d, _mm256_castsi256_pd, __m256i, float64x2_t)
CAST_SIMD_TYPE(__m256i, _mm256_castpd_si256, __m256d, __m128i)




AVX2NEON_ABI
__m128 _mm256_castps256_ps128 (__m256 a)
{
    return a.lo;
}

AVX2NEON_ABI
__m256i _mm256_castsi128_si256 (__m128i a)
{
    __m256i res;
    res.lo = a ;
    res.hi = vdupq_n_s32(0);
    return res;
}

AVX2NEON_ABI
__m128i _mm256_castsi256_si128 (__m256i a)
{
    return a.lo;
}

AVX2NEON_ABI
__m256 _mm256_castps128_ps256 (__m128 a)
{
    __m256 res;
    res.lo = a;
    res.hi = vdupq_n_f32(0);
    return res;
}


AVX2NEON_ABI
__m256 _mm256_broadcast_ss (float const * mem_addr)
{
    __m256 res;
    res.lo = res.hi = vdupq_n_f32(*mem_addr);
    return res;
}


AVX2NEON_ABI
__m256i _mm256_set_epi32 (int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0)
{
    __m256i res;
    res.lo = _mm_set_epi32(e3,e2,e1,e0);
    res.hi = _mm_set_epi32(e7,e6,e5,e4);
    return res;

}

AVX2NEON_ABI
__m256i _mm256_set1_epi32 (int a)
{
    __m256i res;
    res.lo = res.hi = vdupq_n_s32(a);
    return res;
}
AVX2NEON_ABI
__m256i _mm256_set1_epi8 (int a)
{
    __m256i res;
    res.lo = res.hi = vdupq_n_s8(a);
    return res;
}
AVX2NEON_ABI
__m256i _mm256_set1_epi16 (int a)
{
    __m256i res;
    res.lo = res.hi = vdupq_n_s16(a);
    return res;
}




AVX2NEON_ABI
int _mm256_movemask_ps(const __m256& v)
{
    return (_mm_movemask_ps(v.hi) << 4) | _mm_movemask_ps(v.lo);
}

template<int imm8>
AVX2NEON_ABI
__m256 __mm256_permute_ps (const __m256& a)
{
    __m256 res;
    res.lo = _mm_shuffle_ps(a.lo,a.lo,imm8);
    res.hi = _mm_shuffle_ps(a.hi,a.hi,imm8);
    return res;

}

#define _mm256_permute_ps(a,c) __mm256_permute_ps<c>(a)


template<int imm8>
AVX2NEON_ABI
__m256 __mm256_shuffle_ps (const __m256 a,const __m256& b)
{
    __m256 res;
    res.lo = _mm_shuffle_ps(a.lo,b.lo,imm8);
    res.hi = _mm_shuffle_ps(a.hi,b.hi,imm8);
    return res;

}

template<int imm8>
AVX2NEON_ABI
__m256i __mm256_shuffle_epi32 (const __m256i a)
{
    __m256i res;
    res.lo = _mm_shuffle_epi32(a.lo,imm8);
    res.hi = _mm_shuffle_epi32(a.hi,imm8);
    return res;

}

template<int imm8>
AVX2NEON_ABI
__m256i __mm256_srli_si256 (__m256i a)
{
    __m256i res;
    res.lo = _mm_srli_si128(a.lo,imm8);
    res.hi = _mm_srli_si128(a.hi,imm8);
    return res;
}

template<int imm8>
AVX2NEON_ABI
__m256i __mm256_slli_si256 (__m256i a)
{
    __m256i res;
    res.lo = _mm_slli_si128(a.lo,imm8);
    res.hi = _mm_slli_si128(a.hi,imm8);
    return res;
}


#define _mm256_srli_si256(a,b) __mm256_srli_si256<b>(a)
#define _mm256_slli_si256(a,b) __mm256_slli_si256<b>(a)



#define _mm256_shuffle_ps(a,b,c) __mm256_shuffle_ps<c>(a,b)
#define _mm256_shuffle_epi32(a,c) __mm256_shuffle_epi32<c>(a)


AVX2NEON_ABI
__m256i _mm256_set1_epi64x (long long a)
{
    __m256i res;
    int64x2_t t = vdupq_n_s64(a);
    res.lo = res.hi = __m128i(t);
    return res;
}


AVX2NEON_ABI
__m256 _mm256_permute2f128_ps (__m256 a, __m256 b, int imm8)
{
    __m256 res;
    __m128 tmp;
    switch (imm8 & 0x7)
    {
        case 0: tmp = a.lo; break;
        case 1: tmp = a.hi; break;
        case 2: tmp = b.lo; break;
        case 3: tmp = b.hi; break;
    }
    if (imm8 & 0x8)
        tmp = _mm_setzero_ps();



    res.lo = tmp;
    imm8 >>= 4;

    switch (imm8 & 0x7)
    {
        case 0: tmp = a.lo; break;
        case 1: tmp = a.hi; break;
        case 2: tmp = b.lo; break;
        case 3: tmp = b.hi; break;
    }
    if (imm8 & 0x8)
        tmp = _mm_setzero_ps();

    res.hi = tmp;

    return res;
}

AVX2NEON_ABI
__m256 _mm256_moveldup_ps (__m256 a)
{
    __m256 res;
    res.lo = _mm_moveldup_ps(a.lo);
    res.hi = _mm_moveldup_ps(a.hi);
    return res;
}

AVX2NEON_ABI
__m256 _mm256_movehdup_ps (__m256 a)
{
    __m256 res;
    res.lo = _mm_movehdup_ps(a.lo);
    res.hi = _mm_movehdup_ps(a.hi);
    return res;
}

AVX2NEON_ABI
__m256 _mm256_insertf128_ps (__m256 a, __m128 b, int imm8)
{
    __m256 res = a;
    if (imm8 & 1) res.hi = b;
    else res.lo = b;
    return res;
}


AVX2NEON_ABI
__m128 _mm256_extractf128_ps (__m256 a, const int imm8)
{
    if (imm8 & 1) return a.hi;
    return a.lo;
}


AVX2NEON_ABI
__m256d _mm256_movedup_pd (__m256d a)
{
    __m256d res;
    res.lo = _mm_movedup_pd(a.lo);
    res.hi = _mm_movedup_pd(a.hi);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_abs_epi32(__m256i a)
{
   __m256i res;
   res.lo = vabsq_s32(a.lo);
   res.hi = vabsq_s32(a.hi);
   return res;
}

UNARY_AVX_OP(__m256,_mm256_sqrt_ps,_mm_sqrt_ps)
UNARY_AVX_OP(__m256,_mm256_rsqrt_ps,_mm_rsqrt_ps)
UNARY_AVX_OP(__m256,_mm256_rcp_ps,_mm_rcp_ps)
UNARY_AVX_OP(__m256,_mm256_floor_ps,vrndmq_f32)
UNARY_AVX_OP(__m256,_mm256_ceil_ps,vrndpq_f32)
UNARY_AVX_OP(__m256i,_mm256_abs_epi16,_mm_abs_epi16)


BINARY_AVX_OP(__m256i,_mm256_add_epi8,_mm_add_epi8)
BINARY_AVX_OP(__m256i,_mm256_adds_epi8,_mm_adds_epi8)

BINARY_AVX_OP(__m256i,_mm256_hadd_epi32,_mm_hadd_epi32)
BINARY_AVX_OP(__m256i,_mm256_add_epi32,_mm_add_epi32)
BINARY_AVX_OP(__m256i,_mm256_sub_epi32,_mm_sub_epi32)
BINARY_AVX_OP(__m256i,_mm256_mullo_epi32,_mm_mullo_epi32)

BINARY_AVX_OP(__m256i,_mm256_min_epi32,_mm_min_epi32)
BINARY_AVX_OP(__m256i,_mm256_max_epi32,_mm_max_epi32)
BINARY_AVX_OP(__m256i,_mm256_min_epi16,_mm_min_epi16)
BINARY_AVX_OP(__m256i,_mm256_max_epi16,_mm_max_epi16)
BINARY_AVX_OP(__m256i,_mm256_min_epi8,_mm_min_epi8)
BINARY_AVX_OP(__m256i,_mm256_max_epi8,_mm_max_epi8)
BINARY_AVX_OP(__m256i,_mm256_min_epu16,_mm_min_epu16)
BINARY_AVX_OP(__m256i,_mm256_max_epu16,_mm_max_epu16)
BINARY_AVX_OP(__m256i,_mm256_min_epu8,_mm_min_epu8)
BINARY_AVX_OP(__m256i,_mm256_max_epu8,_mm_max_epu8)
BINARY_AVX_OP(__m256i,_mm256_sign_epi16,_mm_sign_epi16)


BINARY_AVX_OP_CAST(__m256i,_mm256_min_epu32,vminq_u32,__m128i,uint32x4_t)
BINARY_AVX_OP_CAST(__m256i,_mm256_max_epu32,vmaxq_u32,__m128i,uint32x4_t)

BINARY_AVX_OP(__m256,_mm256_min_ps,_mm_min_ps)
BINARY_AVX_OP(__m256,_mm256_max_ps,_mm_max_ps)

BINARY_AVX_OP(__m256,_mm256_add_ps,_mm_add_ps)
BINARY_AVX_OP(__m256,_mm256_mul_ps,_mm_mul_ps)
BINARY_AVX_OP(__m256,_mm256_sub_ps,_mm_sub_ps)
BINARY_AVX_OP(__m256,_mm256_div_ps,_mm_div_ps)

BINARY_AVX_OP(__m256,_mm256_and_ps,_mm_and_ps)
BINARY_AVX_OP(__m256,_mm256_andnot_ps,_mm_andnot_ps)
BINARY_AVX_OP(__m256,_mm256_or_ps,_mm_or_ps)
BINARY_AVX_OP(__m256,_mm256_xor_ps,_mm_xor_ps)

BINARY_AVX_OP_CAST(__m256d,_mm256_and_pd,vandq_s64,float64x2_t,int64x2_t)
BINARY_AVX_OP_CAST(__m256d,_mm256_or_pd,vorrq_s64,float64x2_t,int64x2_t)
BINARY_AVX_OP_CAST(__m256d,_mm256_xor_pd,veorq_s64,float64x2_t,int64x2_t)



BINARY_AVX_OP(__m256i,_mm256_and_si256,_mm_and_si128)
BINARY_AVX_OP(__m256i,_mm256_andnot_si256,_mm_andnot_si128)
BINARY_AVX_OP(__m256i,_mm256_or_si256,_mm_or_si128)
BINARY_AVX_OP(__m256i,_mm256_xor_si256,_mm_xor_si128)


BINARY_AVX_OP(__m256,_mm256_unpackhi_ps,_mm_unpackhi_ps)
BINARY_AVX_OP(__m256,_mm256_unpacklo_ps,_mm_unpacklo_ps)
TERNARY_AVX_OP(__m256,_mm256_blendv_ps,_mm_blendv_ps)
TERNARY_AVX_OP(__m256i,_mm256_blendv_epi8,_mm_blendv_epi8)


TERNARY_AVX_OP(__m256,_mm256_fmadd_ps,_mm_fmadd_ps)
TERNARY_AVX_OP(__m256,_mm256_fnmadd_ps,_mm_fnmadd_ps)
TERNARY_AVX_OP(__m256,_mm256_fmsub_ps,_mm_fmsub_ps)
TERNARY_AVX_OP(__m256,_mm256_fnmsub_ps,_mm_fnmsub_ps)



BINARY_AVX_OP(__m256i,_mm256_packs_epi32,_mm_packs_epi32)
BINARY_AVX_OP(__m256i,_mm256_packs_epi16,_mm_packs_epi16)
BINARY_AVX_OP(__m256i,_mm256_packus_epi32,_mm_packus_epi32)
BINARY_AVX_OP(__m256i,_mm256_packus_epi16,_mm_packus_epi16)


BINARY_AVX_OP(__m256i,_mm256_unpackhi_epi64,_mm_unpackhi_epi64)
BINARY_AVX_OP(__m256i,_mm256_unpackhi_epi32,_mm_unpackhi_epi32)
BINARY_AVX_OP(__m256i,_mm256_unpackhi_epi16,_mm_unpackhi_epi16)
BINARY_AVX_OP(__m256i,_mm256_unpackhi_epi8,_mm_unpackhi_epi8)

BINARY_AVX_OP(__m256i,_mm256_unpacklo_epi64,_mm_unpacklo_epi64)
BINARY_AVX_OP(__m256i,_mm256_unpacklo_epi32,_mm_unpacklo_epi32)
BINARY_AVX_OP(__m256i,_mm256_unpacklo_epi16,_mm_unpacklo_epi16)
BINARY_AVX_OP(__m256i,_mm256_unpacklo_epi8,_mm_unpacklo_epi8)

BINARY_AVX_OP(__m256i,_mm256_mulhrs_epi16,_mm_mulhrs_epi16)
BINARY_AVX_OP(__m256i,_mm256_mulhi_epu16,_mm_mulhi_epu16)
BINARY_AVX_OP(__m256i,_mm256_mulhi_epi16,_mm_mulhi_epi16)
//BINARY_AVX_OP(__m256i,_mm256_mullo_epu16,_mm_mullo_epu16)
BINARY_AVX_OP(__m256i,_mm256_mullo_epi16,_mm_mullo_epi16)

BINARY_AVX_OP(__m256i,_mm256_subs_epu16,_mm_subs_epu16)
BINARY_AVX_OP(__m256i,_mm256_adds_epu16,_mm_adds_epu16)
BINARY_AVX_OP(__m256i,_mm256_subs_epi16,_mm_subs_epi16)
BINARY_AVX_OP(__m256i,_mm256_adds_epi16,_mm_adds_epi16)
BINARY_AVX_OP(__m256i,_mm256_sub_epi16,_mm_sub_epi16)
BINARY_AVX_OP(__m256i,_mm256_add_epi16,_mm_add_epi16)
BINARY_AVX_OP(__m256i,_mm256_sub_epi8,_mm_sub_epi8)


BINARY_AVX_OP(__m256i,_mm256_hadd_epi16,_mm_hadd_epi16)
BINARY_AVX_OP(__m256i,_mm256_hadds_epi16,_mm_hadds_epi16)




BINARY_AVX_OP(__m256i,_mm256_cmpeq_epi32,_mm_cmpeq_epi32)
BINARY_AVX_OP(__m256i,_mm256_cmpgt_epi32,_mm_cmpgt_epi32)

BINARY_AVX_OP(__m256i,_mm256_cmpeq_epi8,_mm_cmpeq_epi8)
BINARY_AVX_OP(__m256i,_mm256_cmpgt_epi8,_mm_cmpgt_epi8)

BINARY_AVX_OP(__m256i,_mm256_cmpeq_epi16,_mm_cmpeq_epi16)
BINARY_AVX_OP(__m256i,_mm256_cmpgt_epi16,_mm_cmpgt_epi16)


BINARY_AVX_OP(__m256i,_mm256_shuffle_epi8,_mm_shuffle_epi8)


BINARY_AVX_OP(__m256,_mm256_cmpeq_ps,_mm_cmpeq_ps)
BINARY_AVX_OP(__m256,_mm256_cmpneq_ps,_mm_cmpneq_ps)
BINARY_AVX_OP(__m256,_mm256_cmpnlt_ps,_mm_cmpnlt_ps)
BINARY_AVX_OP(__m256,_mm256_cmpngt_ps,_mm_cmpngt_ps)
BINARY_AVX_OP(__m256,_mm256_cmpge_ps,_mm_cmpge_ps)
BINARY_AVX_OP(__m256,_mm256_cmpnge_ps,_mm_cmpnge_ps)
BINARY_AVX_OP(__m256,_mm256_cmplt_ps,_mm_cmplt_ps)
BINARY_AVX_OP(__m256,_mm256_cmple_ps,_mm_cmple_ps)
BINARY_AVX_OP(__m256,_mm256_cmpgt_ps,_mm_cmpgt_ps)
BINARY_AVX_OP(__m256,_mm256_cmpnle_ps,_mm_cmpnle_ps)


AVX2NEON_ABI
__m256i _mm256_cvtps_epi32 (__m256 a)
{
    __m256i res;
    res.lo = _mm_cvtps_epi32(a.lo);
    res.hi = _mm_cvtps_epi32(a.hi);
    return res;

}

AVX2NEON_ABI
__m256i _mm256_cvttps_epi32 (__m256 a)
{
    __m256i res;
    res.lo = _mm_cvttps_epi32(a.lo);
    res.hi = _mm_cvttps_epi32(a.hi);
    return res;

}

AVX2NEON_ABI
__m256 _mm256_loadu_ps (float const * mem_addr)
{
    __m256 res;
    res.lo = *(__m128 *)(mem_addr + 0);
    res.hi = *(__m128 *)(mem_addr + 4);
    return res;
}
#define _mm256_load_ps _mm256_loadu_ps


AVX2NEON_ABI
int _mm256_testz_ps (const __m256& a, const __m256& b)
{
    __m256 t = a;
    if (&a != &b)
        t = _mm256_and_ps(a,b);

    int32x4_t l  = vshrq_n_s32(vreinterpretq_s32_m128(t.lo),31);
    int32x4_t h  = vshrq_n_s32(vreinterpretq_s32_m128(t.hi),31);
    return vaddvq_s32(vaddq_s32(l,h)) == 0;
}


AVX2NEON_ABI
__m256i _mm256_set_epi64x (int64_t e3, int64_t e2, int64_t e1, int64_t e0)
{
    __m256i res;
    int64x2_t t0 = {e0,e1};
    int64x2_t t1 = {e2,e3};
    res.lo = __m128i(t0);
    res.hi = __m128i(t1);
    return res;
}
AVX2NEON_ABI
__m256i _mm256_setr_epi64x (int64_t e0, int64_t e1, int64_t e2, int64_t e3)
{
    __m256i res;
    int64x2_t t0 = {e0,e1};
    int64x2_t t1 = {e2,e3};
    res.lo = __m128i(t0);
    res.hi = __m128i(t1);
    return res;
}



AVX2NEON_ABI
__m256i _mm256_set_epi8 (char e31, char e30, char e29, char e28, char e27, char e26, char e25, char e24, char e23, char e22, char e21, char e20, char e19, char e18, char e17, char e16, char e15, char e14, char e13, char e12, char e11, char e10, char e9, char e8, char e7, char e6, char e5, char e4, char e3, char e2, char e1, char e0)
{
    int8x16_t lo = {e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15};
    int8x16_t hi = {e16,e17,e18,e19,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31};
    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}

AVX2NEON_ABI
__m256i _mm256_setr_epi8 (char e0, char e1, char e2, char e3, char e4, char e5, char e6, char e7, char e8, char e9, char e10, char e11, char e12, char e13, char e14, char e15, char e16, char e17, char e18, char e19, char e20, char e21, char e22, char e23, char e24, char e25, char e26, char e27, char e28, char e29, char e30, char e31)
{
    int8x16_t lo = {e0,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15};
    int8x16_t hi = {e16,e17,e18,e19,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31};
    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}


AVX2NEON_ABI
__m256i _mm256_set_epi16 (short e15, short e14, short e13, short e12, short e11, short e10, short e9, short e8, short e7, short e6, short e5, short e4, short e3, short e2, short e1, short e0)
{
    int16x8_t lo = {e0,e1,e2,e3,e4,e5,e6,e7};
    int16x8_t hi = {e8,e9,e10,e11,e12,e13,e14,e15};
    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}

AVX2NEON_ABI
__m256i _mm256_setr_epi16 (short e0, short e1, short e2, short e3, short e4, short e5, short e6, short e7, short e8, short e9, short e10, short e11, short e12, short e13, short e14, short e15)
{
    int16x8_t lo = {e0,e1,e2,e3,e4,e5,e6,e7};
    int16x8_t hi = {e8,e9,e10,e11,e12,e13,e14,e15};
    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}




AVX2NEON_ABI
int _mm256_movemask_epi8(const __m256i& a)
{
    return (_mm_movemask_epi8(a.hi) << 16) | _mm_movemask_epi8(a.lo);
}


AVX2NEON_ABI
int _mm256_testz_si256(const __m256i& a,const __m256i& b)
{
    uint32x4_t lo = vandq_u32(a.lo,b.lo);
    uint32x4_t hi = vandq_u32(a.hi,b.hi);

    return (vaddvq_u32(lo) + vaddvq_u32(hi)) == 0;
}

AVX2NEON_ABI
__m256d _mm256_setzero_pd ()
{
    __m256d res;
    res.lo = res.hi = vdupq_n_f64(0);
    return res;
}

AVX2NEON_ABI
int _mm256_movemask_pd (__m256d a)
{
    return (_mm_movemask_pd(a.hi) << 2) | _mm_movemask_pd(a.lo);
}

AVX2NEON_ABI
__m256i _mm256_cmpeq_epi64 (__m256i a, __m256i b)
{
    __m256i res;
    res.lo = _mm_cmpeq_epi64(a.lo, b.lo);
    res.hi = _mm_cmpeq_epi64(a.hi, b.hi);
    return res;
}

AVX2NEON_ABI
__m256d _mm256_cmpeq_pd (__m256d a, __m256d b)
{
    __m256d res;
    res.lo = _mm_cmpeq_pd(a.lo, b.lo);
    res.hi = _mm_cmpeq_pd(a.hi, b.hi);
    return res;
}


AVX2NEON_ABI
int _mm256_testz_pd (const __m256d& a, const __m256d& b)
{
    __m256d t = a;

    if (&a != &b)
        t = _mm256_and_pd(a,b);

    return _mm256_movemask_pd(t) == 0;
}

AVX2NEON_ABI
__m256d _mm256_blendv_pd (__m256d a, __m256d b, __m256d mask)
{
    __m256d res;
    res.lo = _mm_blendv_pd(a.lo, b.lo, mask.lo);
    res.hi = _mm_blendv_pd(a.hi, b.hi, mask.hi);
    return res;
}

template<int imm8>
AVX2NEON_ABI
__m256 __mm256_dp_ps (__m256 a, __m256 b)
{
    __m256 res;
    res.lo = _mm_dp_ps(a.lo, b.lo, imm8);
    res.hi = _mm_dp_ps(a.hi, b.hi, imm8);
    return res;
}

#define _mm256_dp_ps(a,b,c) __mm256_dp_ps<c>(a,b)

AVX2NEON_ABI
double _mm256_permute4x64_pd_select(__m256d a, const int imm8)
{
    switch (imm8 & 3) {
        case 0:
            return ((float64x2_t)a.lo)[0];
        case 1:
            return ((float64x2_t)a.lo)[1];
        case 2:
            return ((float64x2_t)a.hi)[0];
        case 3:
            return ((float64x2_t)a.hi)[1];
    }
    __builtin_unreachable();
    return 0;
}

AVX2NEON_ABI
__m256d _mm256_permute4x64_pd (__m256d a, const int imm8)
{
    float64x2_t lo,hi;
    lo[0] = _mm256_permute4x64_pd_select(a,imm8 >> 0);
    lo[1] = _mm256_permute4x64_pd_select(a,imm8 >> 2);
    hi[0] = _mm256_permute4x64_pd_select(a,imm8 >> 4);
    hi[1] = _mm256_permute4x64_pd_select(a,imm8 >> 6);

    __m256d res;
    res.lo = lo; res.hi = hi;
    return res;
}

AVX2NEON_ABI
__m256i _mm256_insertf128_si256 (__m256i a, __m128i b, int imm8)
{
    return __m256i(_mm256_insertf128_ps((__m256)a,(__m128)b,imm8));
}


AVX2NEON_ABI
__m256i _mm256_loadu_si256 (__m256i const * mem_addr)
{
    __m256i res;
    res.lo = *(__m128i *)((int32_t *)mem_addr + 0);
    res.hi = *(__m128i *)((int32_t *)mem_addr + 4);
    return res;
}

#define _mm256_load_si256 _mm256_loadu_si256

AVX2NEON_ABI
void _mm256_storeu_ps (float * mem_addr, __m256 a)
{
    *(__m128 *)(mem_addr + 0) = a.lo;
    *(__m128 *)(mem_addr + 4) = a.hi;
}

#define _mm256_store_ps _mm256_storeu_ps
#define _mm256_stream_ps _mm256_storeu_ps


AVX2NEON_ABI
void _mm256_storeu_si256 (__m256i * mem_addr, __m256i a)
{
    *(__m128i *)((int32_t *)mem_addr + 0) = a.lo;
    *(__m128i *)((int32_t *)mem_addr + 4) = a.hi;
}

#define _mm256_store_si256 _mm256_storeu_si256



AVX2NEON_ABI
__m256i _mm256_permute4x64_epi64 (const __m256i a, const int imm8)
{
    uint8x16x2_t tbl = {a.lo, a.hi};

    uint8_t sz = sizeof(uint64_t);
    uint8_t u64[4] = {
        (uint8_t)(((imm8 >> 0) & 0x3) * sz),
        (uint8_t)(((imm8 >> 2) & 0x3) * sz),
        (uint8_t)(((imm8 >> 4) & 0x3) * sz),
        (uint8_t)(((imm8 >> 6) & 0x3) * sz),
    };

    uint8x16_t idx_lo = {
        // lo[0] bytes
        (uint8_t)(u64[0]+0), (uint8_t)(u64[0]+1), (uint8_t)(u64[0]+2), (uint8_t)(u64[0]+3),
        (uint8_t)(u64[0]+4), (uint8_t)(u64[0]+5), (uint8_t)(u64[0]+6), (uint8_t)(u64[0]+7),

        // lo[1] bytes
        (uint8_t)(u64[1]+0), (uint8_t)(u64[1]+1), (uint8_t)(u64[1]+2), (uint8_t)(u64[1]+3),
        (uint8_t)(u64[1]+4), (uint8_t)(u64[1]+5), (uint8_t)(u64[1]+6), (uint8_t)(u64[1]+7),
    };
    uint8x16_t idx_hi = {
        // hi[0] bytes
        (uint8_t)(u64[2]+0), (uint8_t)(u64[2]+1), (uint8_t)(u64[2]+2), (uint8_t)(u64[2]+3),
        (uint8_t)(u64[2]+4), (uint8_t)(u64[2]+5), (uint8_t)(u64[2]+6), (uint8_t)(u64[2]+7),

        // hi[1] bytes
        (uint8_t)(u64[3]+0), (uint8_t)(u64[3]+1), (uint8_t)(u64[3]+2), (uint8_t)(u64[3]+3),
        (uint8_t)(u64[3]+4), (uint8_t)(u64[3]+5), (uint8_t)(u64[3]+6), (uint8_t)(u64[3]+7),
    };

    uint8x16_t lo = vqtbl2q_u8(tbl, idx_lo);
    uint8x16_t hi = vqtbl2q_u8(tbl, idx_hi);

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}


AVX2NEON_ABI
__m256i _mm256_permute2x128_si256(const __m256i a,const __m256i b, const int imm8)
{
    return __m256i(_mm256_permute2f128_ps(__m256(a),__m256(b),imm8));
}



AVX2NEON_ABI
__m256 _mm256_maskload_ps (float const * mem_addr, __m256i mask)
{
    __m256 res;
    res.lo = _mm_maskload_ps(mem_addr,mask.lo);
    res.hi = _mm_maskload_ps(mem_addr + 4,mask.hi);
    return res;
}


AVX2NEON_ABI
__m256i _mm256_cvtepu8_epi32 (__m128i a)
{
    uint8x16_t a_u8 = vreinterpretq_u8_m128i(a);     // xxxx xxxx xxxx xxxx HHGG FFEE DDCC BBAA
    uint16x8_t u16x8 = vmovl_u8(vget_low_u8(a_u8));  // 00HH 00GG 00FF 00EE 00DD 00CC 00BB 00AA
    uint32x4_t lo = vmovl_u16(vget_low_u16(u16x8));  // 0000 00DD 0000 00CC 0000 00BB 0000 00AA
    uint32x4_t hi = vmovl_high_u16(u16x8);           // 0000 00HH 0000 00GG 0000 00FF 0000 00EE

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}


AVX2NEON_ABI
__m256i _mm256_cvtepi8_epi32 (__m128i a)
{
    int8x16_t a_s8 = vreinterpretq_s8_m128i(a);     // xxxx xxxx xxxx xxxx HHGG FFEE DDCC BBAA
    int16x8_t s16x8 = vmovl_s8(vget_low_s8(a_s8));  // ssHH ssGG ssFF ssEE ssDD ssCC ssBB ssAA
    int32x4_t lo = vmovl_s16(vget_low_s16(s16x8));  // ssss ssDD ssss ssCC ssss ssBB ssss ssAA
    int32x4_t hi = vmovl_high_s16(s16x8);           // ssss ssHH ssss ssGG ssss ssFF ssss ssEE

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}


AVX2NEON_ABI
__m256i _mm256_cvtepi16_epi32 (__m128i a)
{
    int16x8_t a_s16 = vreinterpretq_s16_m128i(a);   // HHHH GGGG FFFF EEEE DDDD CCCC BBBB AAAA
    int32x4_t lo = vmovl_s16(vget_low_s16(a_s16));  // ssss DDDD ssss CCCC ssss BBBB ssss AAAA
    int32x4_t hi = vmovl_high_s16(a_s16);           // ssss HHHH ssss GGGG ssss FFFF ssss EEEE

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}



AVX2NEON_ABI
void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a)
{
    _mm_maskstore_epi32(mem_addr,mask.lo,a.lo);
    _mm_maskstore_epi32(mem_addr + 4,mask.hi,a.hi);
}

AVX2NEON_ABI
__m256i _mm256_slli_epi64 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_slli_epi64(a.lo,imm8);
    res.hi = _mm_slli_epi64(a.hi,imm8);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_slli_epi32 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_slli_epi32(a.lo,imm8);
    res.hi = _mm_slli_epi32(a.hi,imm8);
    return res;
}


AVX2NEON_ABI
__m256i __mm256_slli_epi16 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_slli_epi16(a.lo,imm8);
    res.hi = _mm_slli_epi16(a.hi,imm8);
    return res;
}


AVX2NEON_ABI
__m256i _mm256_srli_epi32 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_srli_epi32(a.lo,imm8);
    res.hi = _mm_srli_epi32(a.hi,imm8);
    return res;
}

AVX2NEON_ABI
__m256i __mm256_srli_epi16 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_srli_epi16(a.lo,imm8);
    res.hi = _mm_srli_epi16(a.hi,imm8);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_cvtepu16_epi32(__m128i a)
{
    __m256i res;
    res.lo = vmovl_u16(vget_low_u16(a));
    res.hi = vmovl_high_u16(a);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_cvtepu8_epi16(__m128i a)
{
    __m256i res;
    res.lo = vmovl_u8(vget_low_u8(a));
    res.hi = vmovl_high_u8(a);
    return res;
}


AVX2NEON_ABI
__m256i _mm256_srai_epi32 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_srai_epi32(a.lo,imm8);
    res.hi = _mm_srai_epi32(a.hi,imm8);
    return res;
}

AVX2NEON_ABI
__m256i _mm256_srai_epi16 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_srai_epi16(a.lo,imm8);
    res.hi = _mm_srai_epi16(a.hi,imm8);
    return res;
}


AVX2NEON_ABI
__m256i _mm256_sllv_epi32 (__m256i a, __m256i count)
{
    __m256i res;
    res.lo = vshlq_s32(a.lo,count.lo);
    res.hi = vshlq_s32(a.hi,count.hi);
    return res;

}


AVX2NEON_ABI
__m256i _mm256_srav_epi32 (__m256i a, __m256i count)
{
    __m256i res;
    res.lo = vshlq_s32(a.lo,vnegq_s32(count.lo));
    res.hi = vshlq_s32(a.hi,vnegq_s32(count.hi));
    return res;

}

AVX2NEON_ABI
__m256i _mm256_srlv_epi32 (__m256i a, __m256i count)
{
    __m256i res;
    res.lo = __m128i(vshlq_u32(uint32x4_t(a.lo),vnegq_s32(count.lo)));
    res.hi = __m128i(vshlq_u32(uint32x4_t(a.hi),vnegq_s32(count.hi)));
    return res;

}


AVX2NEON_ABI
__m256i _mm256_permute2f128_si256 (__m256i a, __m256i b, int imm8)
{
    return __m256i(_mm256_permute2f128_ps(__m256(a),__m256(b),imm8));
}


AVX2NEON_ABI
__m128i _mm256_extractf128_si256 (__m256i a, const int imm8)
{
    if (imm8 & 1) return a.hi;
    return a.lo;
}

AVX2NEON_ABI
__m256 _mm256_set1_ps(float x)
{
    __m256 res;
    res.lo = res.hi = vdupq_n_f32(x);
    return res;
}

AVX2NEON_ABI
__m256 _mm256_set_ps (float e7, float e6, float e5, float e4, float e3, float e2, float e1, float e0)
{
    __m256 res;
    res.lo = _mm_set_ps(e3,e2,e1,e0);
    res.hi = _mm_set_ps(e7,e6,e5,e4);
    return res;
}

AVX2NEON_ABI
__m256 _mm256_broadcast_ps (__m128 const * mem_addr)
{
    __m256 res;
    res.lo = res.hi = *mem_addr;
    return res;
}

AVX2NEON_ABI
__m256 _mm256_cvtepi32_ps (__m256i a)
{
    __m256 res;
    res.lo = _mm_cvtepi32_ps(a.lo);
    res.hi = _mm_cvtepi32_ps(a.hi);
    return res;
}
AVX2NEON_ABI
void _mm256_maskstore_ps (float * mem_addr, __m256i mask, __m256 a)
{
    uint32x4_t mask_lo = mask.lo;
    uint32x4_t mask_hi = mask.hi;
    float32x4_t a_lo = a.lo;
    float32x4_t a_hi = a.hi;

    for (int i=0;i<4;i++) {
        if (mask_lo[i] & 0x80000000) mem_addr[i] = a_lo[i];
        if (mask_hi[i] & 0x80000000) mem_addr[i+4] = a_hi[i];
    }
}

AVX2NEON_ABI
__m256d _mm256_andnot_pd (__m256d a, __m256d b)
{
    __m256d res;
    res.lo = float64x2_t(_mm_andnot_ps(__m128(a.lo),__m128(b.lo)));
    res.hi = float64x2_t(_mm_andnot_ps(__m128(a.hi),__m128(b.hi)));
    return res;
}

AVX2NEON_ABI
__m256 _mm256_blend_ps (__m256 a, __m256 b, const int imm8)
{
    __m256 res;
    res.lo = _mm_blend_ps(a.lo,b.lo,imm8 & 0xf);
    res.hi = _mm_blend_ps(a.hi,b.hi,imm8 >> 4);
    return res;

}


AVX2NEON_ABI
__m256i _mm256_blend_epi32 (__m256i a, __m256i b, const int imm8)
{
    return __m256i(_mm256_blend_ps(__m256(a),__m256(b),imm8));

}

AVX2NEON_ABI
__m256i _mm256_blend_epi16 (__m256i a, __m256i b, const int imm8)
{
    __m256i res;
    res.lo = _mm_blend_epi16(a.lo,b.lo,imm8);
    res.hi = _mm_blend_epi16(a.hi,b.hi,imm8);
    return res;
}



AVX2NEON_ABI
__m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
{
    int32x4_t vindex_lo = vindex.lo;
    int32x4_t vindex_hi = vindex.hi;
    int32x4_t lo,hi;
    for (int i=0;i<4;i++)
    {
        lo[i] = *(int32_t *)((char *) base_addr + (vindex_lo[i]*scale));
        hi[i] = *(int32_t *)((char *) base_addr + (vindex_hi[i]*scale));
    }

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}


AVX2NEON_ABI
__m256i _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale)
{
    uint32x4_t mask_lo = mask.lo;
    uint32x4_t mask_hi = mask.hi;
    int32x4_t vindex_lo = vindex.lo;
    int32x4_t vindex_hi = vindex.hi;
    int32x4_t lo,hi;
    lo = hi = _mm_setzero_si128();
    for (int i=0;i<4;i++)
    {
        if (mask_lo[i] >> 31) lo[i] = *(int32_t *)((char *) base_addr + (vindex_lo[i]*scale));
        if (mask_hi[i] >> 31) hi[i] = *(int32_t *)((char *) base_addr + (vindex_hi[i]*scale));
    }

    __m256i res;
    res.lo = lo; res.hi = hi;
    return res;
}
