#pragma once

#include "SSE2NEON.h"


#define AVX2NEON_ABI static inline  __attribute__((always_inline))


struct __m256d;

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
__m128 _mm_blend_ps (__m128 a, __m128 b, const int imm8)
{
    __m128 res;
    for (int i=0;i<4;i++)
    {
        if (imm8 & (1<<i))
        {
            res[i] = b[i];
        }
        else{
            res[i] = a[i];
        }
    }
    
    return res;
}

AVX2NEON_ABI
__m128i _mm_blend_epi32 (__m128i a, __m128i b, const int imm8)
{
    __m128i res;
    for (int i=0;i<4;i++)
    {
        if (imm8 & (1<<i))
        {
            res[i] = b[i];
        }
        else{
            res[i] = a[i];
        }
    }
    return res;
}

AVX2NEON_ABI
__m128 _mm_cmpngt_ps (__m128 a, __m128 b)
{
    return __m128(vmvnq_s32(__m128i(_mm_cmpgt_ps(a,b))));
}


AVX2NEON_ABI
__m128i _mm_loadl_epi64 (__m128i const* mem_addr)
{
    int64x2_t y;
    y[0] = *(int64_t *)mem_addr;
    y[1] = 0;
    return __m128i(y);
}

AVX2NEON_ABI
int _mm_movemask_popcnt(__m128 a)
{
    return __builtin_popcount(_mm_movemask_ps(a));
}

AVX2NEON_ABI
__m128 _mm_maskload_ps (float const * mem_addr, __m128i mask)
{
    __m128 res;
    for (int i=0;i<4;i++) {
        if (mask[i] & 0x80000000) res[i] = mem_addr[i]; else res[i] = 0;
    }
    return res;
}

AVX2NEON_ABI
void _mm_maskstore_ps (float * mem_addr, __m128i mask, __m128 a)
{
    for (int i=0;i<4;i++) {
        if (mask[i] & 0x80000000) mem_addr[i] = a[i];
    }
}

AVX2NEON_ABI
void _mm_maskstore_epi32 (int * mem_addr, __m128i mask, __m128i a)
{
    for (int i=0;i<4;i++) {
        if (mask[i] & 0x80000000) mem_addr[i] = a[i];
    }
}

AVX2NEON_ABI
__m128 _mm_fnmsub_ps (__m128 a, __m128 b, __m128 c)
{
    return vnegq_f32(vfmaq_f32(c,a,b));
}

#define _mm_fnmsub_ss _mm_fnmsub_ps

AVX2NEON_ABI
__m128 _mm_fnmadd_ps (__m128 a, __m128 b, __m128 c)
{
    return vfmsq_f32(c,a,b);
}

#define _mm_fnmadd_ss _mm_fnmadd_ps


AVX2NEON_ABI
__m128 _mm_broadcast_ss (float const * mem_addr)
{
    return vdupq_n_f32(*mem_addr);
}


AVX2NEON_ABI
__m128 _mm_fmsub_ps (__m128 a, __m128 b, __m128 c)
{
    return vfmaq_f32(vnegq_f32(c),a,b);
}

#define _mm_fmsub_ss _mm_fmsub_ps
#define _mm_fmadd_ps _mm_madd_ps
#define _mm_fmadd_ss _mm_madd_ps



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
__m128 _mm_cmpnge_ps (__m128 a, __m128 b)
{
    return __m128(vmvnq_s32(__m128i(_mm_cmpge_ps(a,b))));
}


AVX2NEON_ABI
__m128 _mm_permutevar_ps (__m128 a, __m128i b)
{
    __m128 x;
    for (int i=0;i<4;i++)
    {
        x[i] = a[b[i&3]];
    }
    return x;
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

CAST_SIMD_TYPE(__m256d,_mm256_castps_pd,__m256,float64x2_t)
CAST_SIMD_TYPE(__m256i,_mm256_castps_si256,__m256,__m128i)
CAST_SIMD_TYPE(__m256, _mm256_castsi256_ps, __m256i,__m128)
CAST_SIMD_TYPE(__m256, _mm256_castpd_ps ,__m256d,__m128)
CAST_SIMD_TYPE(__m256d, _mm256_castsi256_pd, __m256i,float64x2_t)
CAST_SIMD_TYPE(__m256i, _mm256_castpd_si256, __m256d,__m128i)




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
    __m128i lo = {e0,e1,e2,e3}, hi = {e4,e5,e6,e7};
    __m256i res;
    res.lo = lo; res.hi = hi;
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

#define _mm256_shuffle_ps(a,b,c) __mm256_shuffle_ps<c>(a,b)

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
    res.lo[0] = res.lo[1] = a.lo[0];
    res.lo[2] = res.lo[3] = a.lo[2];
    res.hi[0] = res.hi[1] = a.hi[0];
    res.hi[2] = res.hi[3] = a.hi[2];
    return res;

}

AVX2NEON_ABI
__m256 _mm256_movehdup_ps (__m256 a)
{
    __m256 res;
    res.lo[0] = res.lo[1] = a.lo[1];
    res.lo[2] = res.lo[3] = a.lo[3];
    res.hi[0] = res.hi[1] = a.hi[1];
    res.hi[2] = res.hi[3] = a.hi[3];
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
    res.hi = a.hi;
    res.lo[0] = res.lo[1] = a.lo[0];
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


BINARY_AVX_OP(__m256i,_mm256_add_epi32,_mm_add_epi32)
BINARY_AVX_OP(__m256i,_mm256_sub_epi32,_mm_sub_epi32)
BINARY_AVX_OP(__m256i,_mm256_mullo_epi32,_mm_mullo_epi32)

BINARY_AVX_OP(__m256i,_mm256_min_epi32,_mm_min_epi32)
BINARY_AVX_OP(__m256i,_mm256_max_epi32,_mm_max_epi32)
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
BINARY_AVX_OP(__m256i,_mm256_or_si256,_mm_or_si128)
BINARY_AVX_OP(__m256i,_mm256_xor_si256,_mm_xor_si128)


BINARY_AVX_OP(__m256,_mm256_unpackhi_ps,_mm_unpackhi_ps)
BINARY_AVX_OP(__m256,_mm256_unpacklo_ps,_mm_unpacklo_ps)
TERNARY_AVX_OP(__m256,_mm256_blendv_ps,_mm_blendv_ps)


TERNARY_AVX_OP(__m256,_mm256_fmadd_ps,_mm_fmadd_ps)
TERNARY_AVX_OP(__m256,_mm256_fnmadd_ps,_mm_fnmadd_ps)
TERNARY_AVX_OP(__m256,_mm256_fmsub_ps,_mm_fmsub_ps)
TERNARY_AVX_OP(__m256,_mm256_fnmsub_ps,_mm_fnmsub_ps)


BINARY_AVX_OP(__m256i,_mm256_unpackhi_epi32,_mm_unpackhi_epi32)
BINARY_AVX_OP(__m256i,_mm256_unpacklo_epi32,_mm_unpacklo_epi32)


BINARY_AVX_OP(__m256i,_mm256_cmpeq_epi32,_mm_cmpeq_epi32)
BINARY_AVX_OP(__m256i,_mm256_cmpgt_epi32,_mm_cmpgt_epi32)
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

    __m128i l  = vshrq_n_s32(__m128i(t.lo),31);
    __m128i h  = vshrq_n_s32(__m128i(t.hi),31);
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
__m256d _mm256_setzero_pd ()
{
    __m256d res;
    res.lo = res.hi = vdupq_n_f64(0);
    return res;
}

AVX2NEON_ABI
int _mm256_movemask_pd (__m256d a)
{
    int res = 0;
    uint64x2_t x;
    x = uint64x2_t(a.lo);
    res |= (x[0] >> 63) ? 1 : 0;
    res |= (x[0] >> 63) ? 2 : 0;
    x = uint64x2_t(a.hi);
    res |= (x[0] >> 63) ? 4 : 0;
    res |= (x[0] >> 63) ? 8 : 0;
    return res;
}

AVX2NEON_ABI
__m256i _mm256_cmpeq_epi64 (__m256i a, __m256i b)
{
    __m256i res;
    res.lo = __m128i(vceqq_s64(int64x2_t(a.lo),int64x2_t(b.lo)));
    res.hi = __m128i(vceqq_s64(int64x2_t(a.hi),int64x2_t(b.hi)));
    return res;
}

AVX2NEON_ABI
__m256i _mm256_cmpeq_pd (__m256d a, __m256d b)
{
    __m256i res;
    res.lo = __m128i(vceqq_f64(a.lo,b.lo));
    res.hi = __m128i(vceqq_f64(a.hi,b.hi));
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
    uint64x2_t t = uint64x2_t(mask.lo);
    res.lo[0] = (t[0] >> 63) ? b.lo[0] : a.lo[0];
    res.lo[1] = (t[1] >> 63) ? b.lo[1] : a.lo[1];
    t = uint64x2_t(mask.hi);
    res.hi[0] = (t[0] >> 63) ? b.hi[0] : a.hi[0];
    res.hi[1] = (t[1] >> 63) ? b.hi[1] : a.hi[1];
    return res;
}

template<int imm8>
__m256 __mm256_dp_ps (__m256 a, __m256 b)
{
    __m256 res;
    res.lo = _mm_dp_ps(a.lo,b.lo,imm8);
    res.hi = _mm_dp_ps(a.hi,b.hi,imm8);
    return res;
}

#define _mm256_dp_ps(a,b,c) __mm256_dp_ps<c>(a,b)

AVX2NEON_ABI
double _mm256_permute4x64_pd_select(__m256d a, const int imm8)
{
    switch (imm8 & 3) {
        case 0:
            return a.lo[0];
        case 1:
            return a.lo[1];
        case 2:
            return a.hi[0];
        case 3:
            return a.hi[1];
    }
    __builtin_unreachable();
    return 0;
}

AVX2NEON_ABI
__m256d _mm256_permute4x64_pd (__m256d a, const int imm8)
{
    __m256d res;
    res.lo[0] = _mm256_permute4x64_pd_select(a,imm8 >> 0);
    res.lo[1] = _mm256_permute4x64_pd_select(a,imm8 >> 2);
    res.hi[0] = _mm256_permute4x64_pd_select(a,imm8 >> 4);
    res.hi[1] = _mm256_permute4x64_pd_select(a,imm8 >> 6);
    
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
    *(__m128i *)((int *)mem_addr + 0) = a.lo;
    *(__m128i *)((int *)mem_addr + 4) = a.hi;

}

#define _mm256_store_si256 _mm256_storeu_si256



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
    __m256i res;
    uint8x16_t x = uint8x16_t(a);
    for (int i=0;i<4;i++)
    {
        res.lo[i] = x[i];
        res.hi[i] = x[i+4];
    }
    return res;
}


AVX2NEON_ABI
__m256i _mm256_cvtepi8_epi32 (__m128i a)
{
    __m256i res;
    int8x16_t x = int8x16_t(a);
    for (int i=0;i<4;i++)
    {
        res.lo[i] = x[i];
        res.hi[i] = x[i+4];
    }
    return res;
}


AVX2NEON_ABI
__m256i _mm256_cvtepu16_epi32 (__m128i a)
{
    __m256i res;
    uint16x8_t x = uint16x8_t(a);
    for (int i=0;i<4;i++)
    {
        res.lo[i] = x[i];
        res.hi[i] = x[i+4];
    }
    return res;
}

AVX2NEON_ABI
__m256i _mm256_cvtepi16_epi32 (__m128i a)
{
    __m256i res;
    int16x8_t x = int16x8_t(a);
    for (int i=0;i<4;i++)
    {
        res.lo[i] = x[i];
        res.hi[i] = x[i+4];
    }
    return res;
}



AVX2NEON_ABI
void _mm256_maskstore_epi32 (int* mem_addr, __m256i mask, __m256i a)
{
    _mm_maskstore_epi32(mem_addr,mask.lo,a.lo);
    _mm_maskstore_epi32(mem_addr + 4,mask.hi,a.hi);
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
__m256i _mm256_srli_epi32 (__m256i a, int imm8)
{
    __m256i res;
    res.lo = _mm_srli_epi32(a.lo,imm8);
    res.hi = _mm_srli_epi32(a.hi,imm8);
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
    for (int i=0;i<4;i++) {
        if (mask.lo[i] & 0x80000000) mem_addr[i] = a.lo[i];
        if (mask.hi[i] & 0x80000000) mem_addr[i+4] = a.hi[i];
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
    __m256i res;
    res.lo = _mm_blend_epi32(a.lo,b.lo,imm8 & 0xf);
    res.hi = _mm_blend_epi32(a.hi,b.hi,imm8 >> 4);
    return res;

}

AVX2NEON_ABI
__m256i _mm256_i32gather_epi32 (int const* base_addr, __m256i vindex, const int scale)
{
    __m256i res;
    for (int i=0;i<4;i++)
    {
        res.lo[i] = *(int *)((char *) base_addr + (vindex.lo[i]*scale));
        res.hi[i] = *(int *)((char *) base_addr + (vindex.hi[i]*scale));
    }
    return res;
}


AVX2NEON_ABI
__m256i _mm256_mask_i32gather_epi32 (__m256i src, int const* base_addr, __m256i vindex, __m256i mask, const int scale)
{
    __m256i res = _mm256_setzero_si256();
    for (int i=0;i<4;i++)
    {
        if (mask.lo[i] >> 31) res.lo[i] = *(int *)((char *) base_addr + (vindex.lo[i]*scale));
        if (mask.hi[i] >> 31) res.hi[i] = *(int *)((char *) base_addr + (vindex.hi[i]*scale));
    }
    
    return res;

}


