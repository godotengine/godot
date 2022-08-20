// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define vboolf vboolf_impl
#define vboold vboold_impl
#define vint vint_impl
#define vuint vuint_impl
#define vllong vllong_impl
#define vfloat vfloat_impl
#define vdouble vdouble_impl

namespace embree
{
  /* 8-wide AVX float type */
  template<>
  struct vfloat<8>
  {
    ALIGNED_STRUCT_(32);
   
    typedef vboolf8 Bool;
    typedef vint8   Int;
    typedef vfloat8 Float;

    enum  { size = 8 };                        // number of SIMD elements
    union { __m256 v; float f[8]; int i[8]; }; // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vfloat() {}
    __forceinline vfloat(const vfloat8& other) { v = other.v; }
    __forceinline vfloat8& operator =(const vfloat8& other) { v = other.v; return *this; }

    __forceinline vfloat(__m256 a) : v(a) {}
    __forceinline operator const __m256&() const { return v; }
    __forceinline operator       __m256&()       { return v; }

    __forceinline explicit vfloat(const vfloat4& a) : v(_mm256_insertf128_ps(_mm256_castps128_ps256(a),a,1)) {}
    __forceinline vfloat(const vfloat4& a, const vfloat4& b) : v(_mm256_insertf128_ps(_mm256_castps128_ps256(a),b,1)) {}

    __forceinline explicit vfloat(const char* a) : v(_mm256_loadu_ps((const float*)a)) {}
    __forceinline vfloat(float a) : v(_mm256_set1_ps(a)) {}
    __forceinline vfloat(float a, float b) : v(_mm256_set_ps(b, a, b, a, b, a, b, a)) {}
    __forceinline vfloat(float a, float b, float c, float d) : v(_mm256_set_ps(d, c, b, a, d, c, b, a)) {}
    __forceinline vfloat(float a, float b, float c, float d, float e, float f, float g, float h) : v(_mm256_set_ps(h, g, f, e, d, c, b, a)) {}

    __forceinline explicit vfloat(__m256i a) : v(_mm256_cvtepi32_ps(a)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vfloat(ZeroTy)   : v(_mm256_setzero_ps()) {}
    __forceinline vfloat(OneTy)    : v(_mm256_set1_ps(1.0f)) {}
    __forceinline vfloat(PosInfTy) : v(_mm256_set1_ps(pos_inf)) {}
    __forceinline vfloat(NegInfTy) : v(_mm256_set1_ps(neg_inf)) {}
    __forceinline vfloat(StepTy)   : v(_mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f)) {}
    __forceinline vfloat(NaNTy)    : v(_mm256_set1_ps(nan)) {}
    __forceinline vfloat(UndefinedTy) : v(_mm256_undefined_ps()) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline vfloat8 broadcast(const void* a) {
      return _mm256_broadcast_ss((float*)a); 
    }

    static __forceinline vfloat8 load(const char* ptr) {
#if defined(__AVX2__)
      return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadu_si128((__m128i*)ptr)));
#else
      return vfloat8(vfloat4::load(ptr),vfloat4::load(ptr+4));
#endif
    }

    static __forceinline vfloat8 load(const unsigned char* ptr) {
#if defined(__AVX2__)
      return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ptr)));
#else
      return vfloat8(vfloat4::load(ptr),vfloat4::load(ptr+4));
#endif
    }

    static __forceinline vfloat8 load(const short* ptr) {
#if defined(__AVX2__)
      return _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128((__m128i*)ptr)));
#else
      return vfloat8(vfloat4::load(ptr),vfloat4::load(ptr+4));
#endif
    }
      
    static __forceinline vfloat8 load (const void* ptr) { return _mm256_load_ps((float*)ptr); }
    static __forceinline vfloat8 loadu(const void* ptr) { return _mm256_loadu_ps((float*)ptr); }

    static __forceinline void store (void* ptr, const vfloat8& v) { return _mm256_store_ps((float*)ptr,v); }
    static __forceinline void storeu(void* ptr, const vfloat8& v) { return _mm256_storeu_ps((float*)ptr,v); }

#if defined(__AVX512VL__)

    static __forceinline vfloat8 load (const vboolf8& mask, const void* ptr) { return _mm256_mask_load_ps (_mm256_setzero_ps(),mask,(float*)ptr); }
    static __forceinline vfloat8 loadu(const vboolf8& mask, const void* ptr) { return _mm256_mask_loadu_ps(_mm256_setzero_ps(),mask,(float*)ptr); }

    static __forceinline void store (const vboolf8& mask, void* ptr, const vfloat8& v) { _mm256_mask_store_ps ((float*)ptr,mask,v); }
    static __forceinline void storeu(const vboolf8& mask, void* ptr, const vfloat8& v) { _mm256_mask_storeu_ps((float*)ptr,mask,v); }
#else
    static __forceinline vfloat8 load (const vboolf8& mask, const void* ptr) { return _mm256_maskload_ps((float*)ptr,(__m256i)mask); }
    static __forceinline vfloat8 loadu(const vboolf8& mask, const void* ptr) { return _mm256_maskload_ps((float*)ptr,(__m256i)mask); }

    static __forceinline void store (const vboolf8& mask, void* ptr, const vfloat8& v) { _mm256_maskstore_ps((float*)ptr,(__m256i)mask,v); }
    static __forceinline void storeu(const vboolf8& mask, void* ptr, const vfloat8& v) { _mm256_maskstore_ps((float*)ptr,(__m256i)mask,v); }
#endif
    
#if defined(__AVX2__)
    static __forceinline vfloat8 load_nt(void* ptr) {
      return _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i*)ptr));
    }
#endif
    
    static __forceinline void store_nt(void* ptr, const vfloat8& v) {
      _mm256_stream_ps((float*)ptr,v);
    }

    template<int scale = 4>
    static __forceinline vfloat8 gather(const float* ptr, const vint8& index) {
#if defined(__AVX2__)
      return _mm256_i32gather_ps(ptr, index ,scale);
#else
      return vfloat8(
          *(float*)(((char*)ptr)+scale*index[0]),
          *(float*)(((char*)ptr)+scale*index[1]),
          *(float*)(((char*)ptr)+scale*index[2]),
          *(float*)(((char*)ptr)+scale*index[3]),
          *(float*)(((char*)ptr)+scale*index[4]),
          *(float*)(((char*)ptr)+scale*index[5]),
          *(float*)(((char*)ptr)+scale*index[6]),
          *(float*)(((char*)ptr)+scale*index[7]));
#endif
    }

    template<int scale = 4>
    static __forceinline vfloat8 gather(const vboolf8& mask, const float* ptr, const vint8& index) {
      vfloat8 r = zero;
#if defined(__AVX512VL__)
      return _mm256_mmask_i32gather_ps(r, mask, index, ptr, scale);
#elif defined(__AVX2__)
      return _mm256_mask_i32gather_ps(r, ptr, index, mask, scale);
#else
      if (likely(mask[0])) r[0] = *(float*)(((char*)ptr)+scale*index[0]);
      if (likely(mask[1])) r[1] = *(float*)(((char*)ptr)+scale*index[1]);
      if (likely(mask[2])) r[2] = *(float*)(((char*)ptr)+scale*index[2]);
      if (likely(mask[3])) r[3] = *(float*)(((char*)ptr)+scale*index[3]);
      if (likely(mask[4])) r[4] = *(float*)(((char*)ptr)+scale*index[4]);
      if (likely(mask[5])) r[5] = *(float*)(((char*)ptr)+scale*index[5]);
      if (likely(mask[6])) r[6] = *(float*)(((char*)ptr)+scale*index[6]);
      if (likely(mask[7])) r[7] = *(float*)(((char*)ptr)+scale*index[7]);
      return r;
    #endif
    }

    template<int scale = 4>
    static __forceinline void scatter(void* ptr, const vint8& ofs, const vfloat8& v)
    {
#if defined(__AVX512VL__)
      _mm256_i32scatter_ps((float*)ptr, ofs, v, scale);
#else
      *(float*)(((char*)ptr)+scale*ofs[0]) = v[0];
      *(float*)(((char*)ptr)+scale*ofs[1]) = v[1];
      *(float*)(((char*)ptr)+scale*ofs[2]) = v[2];
      *(float*)(((char*)ptr)+scale*ofs[3]) = v[3];
      *(float*)(((char*)ptr)+scale*ofs[4]) = v[4];
      *(float*)(((char*)ptr)+scale*ofs[5]) = v[5];
      *(float*)(((char*)ptr)+scale*ofs[6]) = v[6];
      *(float*)(((char*)ptr)+scale*ofs[7]) = v[7];
#endif
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf8& mask, void* ptr, const vint8& ofs, const vfloat8& v)
    {
#if defined(__AVX512VL__)
      _mm256_mask_i32scatter_ps((float*)ptr, mask, ofs, v, scale);
#else
      if (likely(mask[0])) *(float*)(((char*)ptr)+scale*ofs[0]) = v[0];
      if (likely(mask[1])) *(float*)(((char*)ptr)+scale*ofs[1]) = v[1];
      if (likely(mask[2])) *(float*)(((char*)ptr)+scale*ofs[2]) = v[2];
      if (likely(mask[3])) *(float*)(((char*)ptr)+scale*ofs[3]) = v[3];
      if (likely(mask[4])) *(float*)(((char*)ptr)+scale*ofs[4]) = v[4];
      if (likely(mask[5])) *(float*)(((char*)ptr)+scale*ofs[5]) = v[5];
      if (likely(mask[6])) *(float*)(((char*)ptr)+scale*ofs[6]) = v[6];
      if (likely(mask[7])) *(float*)(((char*)ptr)+scale*ofs[7]) = v[7];
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const float& operator [](size_t index) const { assert(index < 8); return f[index]; }
    __forceinline       float& operator [](size_t index)       { assert(index < 8); return f[index]; }
  };


  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 asFloat(const vint8&   a) { return _mm256_castsi256_ps(a); }
  __forceinline vint8   asInt  (const vfloat8& a) { return _mm256_castps_si256(a); }

  __forceinline vint8   toInt  (const vfloat8& a) { return vint8(a); }
  __forceinline vfloat8 toFloat(const vint8&   a) { return vfloat8(a); }

  __forceinline vfloat8 operator +(const vfloat8& a) { return a; }
  __forceinline vfloat8 operator -(const vfloat8& a) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000)); 
    return _mm256_xor_ps(a, mask);
  }
  __forceinline vfloat8 abs(const vfloat8& a) {
    const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    return _mm256_and_ps(a, mask);
  }
  __forceinline vfloat8 sign   (const vfloat8& a) { return _mm256_blendv_ps(vfloat8(one), -vfloat8(one), _mm256_cmp_ps(a, vfloat8(zero), _CMP_NGE_UQ)); }
  __forceinline vfloat8 signmsk(const vfloat8& a) { return _mm256_and_ps(a,_mm256_castsi256_ps(_mm256_set1_epi32(0x80000000))); }


  static __forceinline vfloat8 rcp(const vfloat8& a)
  {
#if defined(__AVX512VL__)
    const vfloat8 r = _mm256_rcp14_ps(a);
#else
    const vfloat8 r = _mm256_rcp_ps(a);
#endif

#if defined(__AVX2__)
    return _mm256_mul_ps(r, _mm256_fnmadd_ps(r, a, vfloat8(2.0f)));
#else
    return _mm256_mul_ps(r, _mm256_sub_ps(vfloat8(2.0f), _mm256_mul_ps(r, a)));
#endif
  }
  __forceinline vfloat8 sqr (const vfloat8& a) { return _mm256_mul_ps(a,a); }
  __forceinline vfloat8 sqrt(const vfloat8& a) { return _mm256_sqrt_ps(a); }

  static __forceinline vfloat8 rsqrt(const vfloat8& a)
  {
#if defined(__AVX512VL__)
    const vfloat8 r = _mm256_rsqrt14_ps(a);
#else
    const vfloat8 r = _mm256_rsqrt_ps(a);
#endif

#if defined(__AVX2__)
    return _mm256_fmadd_ps(_mm256_set1_ps(1.5f), r,
                           _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, _mm256_set1_ps(-0.5f)), r), _mm256_mul_ps(r, r))); 
#else
    return _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(1.5f), r),
                         _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(a, _mm256_set1_ps(-0.5f)), r), _mm256_mul_ps(r, r)));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 operator +(const vfloat8& a, const vfloat8& b) { return _mm256_add_ps(a, b); }
  __forceinline vfloat8 operator +(const vfloat8& a, float          b) { return a + vfloat8(b); }
  __forceinline vfloat8 operator +(float          a, const vfloat8& b) { return vfloat8(a) + b; }

  __forceinline vfloat8 operator -(const vfloat8& a, const vfloat8& b) { return _mm256_sub_ps(a, b); }
  __forceinline vfloat8 operator -(const vfloat8& a, float          b) { return a - vfloat8(b); }
  __forceinline vfloat8 operator -(float          a, const vfloat8& b) { return vfloat8(a) - b; }

  __forceinline vfloat8 operator *(const vfloat8& a, const vfloat8& b) { return _mm256_mul_ps(a, b); }
  __forceinline vfloat8 operator *(const vfloat8& a, float          b) { return a * vfloat8(b); }
  __forceinline vfloat8 operator *(float          a, const vfloat8& b) { return vfloat8(a) * b; }

  __forceinline vfloat8 operator /(const vfloat8& a, const vfloat8& b) { return _mm256_div_ps(a, b); }
  __forceinline vfloat8 operator /(const vfloat8& a, float          b) { return a / vfloat8(b); }
  __forceinline vfloat8 operator /(float          a, const vfloat8& b) { return vfloat8(a) / b; }

  __forceinline vfloat8 operator &(const vfloat8& a, const vfloat8& b) { return _mm256_and_ps(a,b); }
  __forceinline vfloat8 operator |(const vfloat8& a, const vfloat8& b) { return _mm256_or_ps(a,b); }
  __forceinline vfloat8 operator ^(const vfloat8& a, const vfloat8& b) { return _mm256_xor_ps(a,b); }
  __forceinline vfloat8 operator ^(const vfloat8& a, const vint8&   b) { return _mm256_xor_ps(a,_mm256_castsi256_ps(b)); }

  __forceinline vfloat8 min(const vfloat8& a, const vfloat8& b) { return _mm256_min_ps(a, b); }
  __forceinline vfloat8 min(const vfloat8& a, float          b) { return _mm256_min_ps(a, vfloat8(b)); }
  __forceinline vfloat8 min(float          a, const vfloat8& b) { return _mm256_min_ps(vfloat8(a), b); }

  __forceinline vfloat8 max(const vfloat8& a, const vfloat8& b) { return _mm256_max_ps(a, b); }
  __forceinline vfloat8 max(const vfloat8& a, float          b) { return _mm256_max_ps(a, vfloat8(b)); }
  __forceinline vfloat8 max(float          a, const vfloat8& b) { return _mm256_max_ps(vfloat8(a), b); }

  /* need "static __forceinline for MSVC, otherwise we'll link the wrong version in debug mode */
#if defined(__AVX2__)

  static __forceinline vfloat8 mini(const vfloat8& a, const vfloat8& b) {
    const vint8 ai = _mm256_castps_si256(a);
    const vint8 bi = _mm256_castps_si256(b);
    const vint8 ci = _mm256_min_epi32(ai,bi);
    return _mm256_castsi256_ps(ci);
  }

  static __forceinline vfloat8 maxi(const vfloat8& a, const vfloat8& b) {
    const vint8 ai = _mm256_castps_si256(a);
    const vint8 bi = _mm256_castps_si256(b);
    const vint8 ci = _mm256_max_epi32(ai,bi);
    return _mm256_castsi256_ps(ci);
  }

  static __forceinline vfloat8 minui(const vfloat8& a, const vfloat8& b) {
    const vint8 ai = _mm256_castps_si256(a);
    const vint8 bi = _mm256_castps_si256(b);
    const vint8 ci = _mm256_min_epu32(ai,bi);
    return _mm256_castsi256_ps(ci);
  }

  static __forceinline vfloat8 maxui(const vfloat8& a, const vfloat8& b) {
    const vint8 ai = _mm256_castps_si256(a);
    const vint8 bi = _mm256_castps_si256(b);
    const vint8 ci = _mm256_max_epu32(ai,bi);
    return _mm256_castsi256_ps(ci);
  }

#else

  static __forceinline vfloat8 mini(const vfloat8& a, const vfloat8& b) {
    return asFloat(min(asInt(a),asInt(b)));
  }

  static __forceinline vfloat8 maxi(const vfloat8& a, const vfloat8& b) {
    return asFloat(max(asInt(a),asInt(b)));
  }

#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  static __forceinline vfloat8 madd  (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return _mm256_fmadd_ps(a,b,c); }
  static __forceinline vfloat8 msub  (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return _mm256_fmsub_ps(a,b,c); }
  static __forceinline vfloat8 nmadd (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return _mm256_fnmadd_ps(a,b,c); }
  static __forceinline vfloat8 nmsub (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return _mm256_fnmsub_ps(a,b,c); }
#else
  static __forceinline vfloat8 madd  (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return a*b+c; }
  static __forceinline vfloat8 msub  (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return a*b-c; }
  static __forceinline vfloat8 nmadd (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return -a*b+c;}
  static __forceinline vfloat8 nmsub (const vfloat8& a, const vfloat8& b, const vfloat8& c) { return -a*b-c; }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8& operator +=(vfloat8& a, const vfloat8& b) { return a = a + b; }
  __forceinline vfloat8& operator +=(vfloat8& a, float          b) { return a = a + b; }

  __forceinline vfloat8& operator -=(vfloat8& a, const vfloat8& b) { return a = a - b; }
  __forceinline vfloat8& operator -=(vfloat8& a, float          b) { return a = a - b; }

  __forceinline vfloat8& operator *=(vfloat8& a, const vfloat8& b) { return a = a * b; }
  __forceinline vfloat8& operator *=(vfloat8& a, float          b) { return a = a * b; }

  __forceinline vfloat8& operator /=(vfloat8& a, const vfloat8& b) { return a = a / b; }
  __forceinline vfloat8& operator /=(vfloat8& a, float          b) { return a = a / b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  static __forceinline vboolf8 operator ==(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_EQ); }
  static __forceinline vboolf8 operator !=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_NE); }
  static __forceinline vboolf8 operator < (const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_LT); }
  static __forceinline vboolf8 operator >=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_GE); }
  static __forceinline vboolf8 operator > (const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_GT); }
  static __forceinline vboolf8 operator <=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps_mask(a, b, _MM_CMPINT_LE); }

  static __forceinline vfloat8 select(const vboolf8& m, const vfloat8& t, const vfloat8& f) {
    return _mm256_mask_blend_ps(m, f, t);
  }
#else
  static __forceinline vboolf8 operator ==(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);  }
  static __forceinline vboolf8 operator !=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_NEQ_UQ); }
  static __forceinline vboolf8 operator < (const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_LT_OS);  }
  static __forceinline vboolf8 operator >=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_NLT_US); }
  static __forceinline vboolf8 operator > (const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_NLE_US); }
  static __forceinline vboolf8 operator <=(const vfloat8& a, const vfloat8& b) { return _mm256_cmp_ps(a, b, _CMP_LE_OS);  }

  static __forceinline vfloat8 select(const vboolf8& m, const vfloat8& t, const vfloat8& f) {
    return _mm256_blendv_ps(f, t, m); 
  }
#endif

  template<int mask>
    __forceinline vfloat8 select(const vfloat8& t, const vfloat8& f) {
    return _mm256_blend_ps(f, t, mask);
  }

  __forceinline vboolf8 operator ==(const vfloat8& a, const float&   b) { return a == vfloat8(b); }
  __forceinline vboolf8 operator ==(const float&   a, const vfloat8& b) { return vfloat8(a) == b; }

  __forceinline vboolf8 operator !=(const vfloat8& a, const float&   b) { return a != vfloat8(b); }
  __forceinline vboolf8 operator !=(const float&   a, const vfloat8& b) { return vfloat8(a) != b; }

  __forceinline vboolf8 operator < (const vfloat8& a, const float&   b) { return a <  vfloat8(b); }
  __forceinline vboolf8 operator < (const float&   a, const vfloat8& b) { return vfloat8(a) <  b; }

  __forceinline vboolf8 operator >=(const vfloat8& a, const float&   b) { return a >= vfloat8(b); }
  __forceinline vboolf8 operator >=(const float&   a, const vfloat8& b) { return vfloat8(a) >= b; }

  __forceinline vboolf8 operator > (const vfloat8& a, const float&   b) { return a >  vfloat8(b); }
  __forceinline vboolf8 operator > (const float&   a, const vfloat8& b) { return vfloat8(a) >  b; }

  __forceinline vboolf8 operator <=(const vfloat8& a, const float&   b) { return a <= vfloat8(b); }
  __forceinline vboolf8 operator <=(const float&   a, const vfloat8& b) { return vfloat8(a) <= b; }

  __forceinline vboolf8 eq(const vfloat8& a, const vfloat8& b) { return a == b; }
  __forceinline vboolf8 ne(const vfloat8& a, const vfloat8& b) { return a != b; }
  __forceinline vboolf8 lt(const vfloat8& a, const vfloat8& b) { return a <  b; }
  __forceinline vboolf8 ge(const vfloat8& a, const vfloat8& b) { return a >= b; }
  __forceinline vboolf8 gt(const vfloat8& a, const vfloat8& b) { return a >  b; }
  __forceinline vboolf8 le(const vfloat8& a, const vfloat8& b) { return a <= b; }

#if defined(__AVX512VL__)
  static __forceinline vboolf8 eq(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_EQ); }
  static __forceinline vboolf8 ne(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_NE); }
  static __forceinline vboolf8 lt(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_LT); }
  static __forceinline vboolf8 ge(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_GE); }
  static __forceinline vboolf8 gt(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_GT); }
  static __forceinline vboolf8 le(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return _mm256_mask_cmp_ps_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  static __forceinline vboolf8 eq(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a == b); }
  static __forceinline vboolf8 ne(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a != b); }
  static __forceinline vboolf8 lt(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a <  b); }
  static __forceinline vboolf8 ge(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a >= b); }
  static __forceinline vboolf8 gt(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a >  b); }
  static __forceinline vboolf8 le(const vboolf8& mask, const vfloat8& a, const vfloat8& b) { return mask & (a <= b); }
#endif

  __forceinline vfloat8 lerp(const vfloat8& a, const vfloat8& b, const vfloat8& t) {
    return madd(t,b-a,a);
  }

  __forceinline bool isvalid (const vfloat8& v) {
    return all((v > vfloat8(-FLT_LARGE)) & (v < vfloat8(+FLT_LARGE)));
  }

  __forceinline bool is_finite (const vfloat8& a) {
    return all((a >= vfloat8(-FLT_MAX)) & (a <= vfloat8(+FLT_MAX)));
  }

  __forceinline bool is_finite (const vboolf8& valid, const vfloat8& a) {
    return all(valid, (a >= vfloat8(-FLT_MAX)) & (a <= vfloat8(+FLT_MAX)));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Rounding Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 floor(const vfloat8& a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEG_INF    ); }
  __forceinline vfloat8 ceil (const vfloat8& a) { return _mm256_round_ps(a, _MM_FROUND_TO_POS_INF    ); }
  __forceinline vfloat8 trunc(const vfloat8& a) { return _mm256_round_ps(a, _MM_FROUND_TO_ZERO       ); }
  __forceinline vfloat8 round(const vfloat8& a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT); }
  __forceinline vfloat8 frac (const vfloat8& a) { return a-floor(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 unpacklo(const vfloat8& a, const vfloat8& b) { return _mm256_unpacklo_ps(a, b); }
  __forceinline vfloat8 unpackhi(const vfloat8& a, const vfloat8& b) { return _mm256_unpackhi_ps(a, b); }

  template<int i>
  __forceinline vfloat8 shuffle(const vfloat8& v) {
    return _mm256_permute_ps(v, _MM_SHUFFLE(i, i, i, i));
  }

  template<int i0, int i1>
  __forceinline vfloat8 shuffle4(const vfloat8& v) {
    return _mm256_permute2f128_ps(v, v, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1>
  __forceinline vfloat8 shuffle4(const vfloat8& a, const vfloat8& b) {
    return _mm256_permute2f128_ps(a, b, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vfloat8 shuffle(const vfloat8& v) {
    return _mm256_permute_ps(v, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vfloat8 shuffle(const vfloat8& a, const vfloat8& b) {
    return _mm256_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<> __forceinline vfloat8 shuffle<0, 0, 2, 2>(const vfloat8& v) { return _mm256_moveldup_ps(v); }
  template<> __forceinline vfloat8 shuffle<1, 1, 3, 3>(const vfloat8& v) { return _mm256_movehdup_ps(v); }
  template<> __forceinline vfloat8 shuffle<0, 1, 0, 1>(const vfloat8& v) { return _mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(v))); }

  __forceinline vfloat8 broadcast(const float* ptr) { return _mm256_broadcast_ss(ptr); }
  template<size_t i> __forceinline vfloat8 insert4(const vfloat8& a, const vfloat4& b) { return _mm256_insertf128_ps(a, b, i); }
  template<size_t i> __forceinline vfloat4 extract4   (const vfloat8& a) { return _mm256_extractf128_ps(a, i); }
  template<>         __forceinline vfloat4 extract4<0>(const vfloat8& a) { return _mm256_castps256_ps128(a);   }

  __forceinline float toScalar(const vfloat8& v) { return _mm_cvtss_f32(_mm256_castps256_ps128(v)); }

#if defined (__AVX2__)
  static __forceinline vfloat8 permute(const vfloat8& a, const __m256i& index) {
    return _mm256_permutevar8x32_ps(a, index);
  }
#endif

#if defined(__AVX512VL__)
  template<int i>
  static __forceinline vfloat8 align_shift_right(const vfloat8& a, const vfloat8& b) {
    return _mm256_castsi256_ps(_mm256_alignr_epi32(_mm256_castps_si256(a), _mm256_castps_si256(b), i));
  }  
#endif

#if defined (__AVX_I__)
  template<const int mode>
  static __forceinline vint4 convert_to_hf16(const vfloat8& a) {
    return _mm256_cvtps_ph(a, mode);
  }

  static __forceinline vfloat8 convert_from_hf16(const vint4& a) {
    return _mm256_cvtph_ps(a);
  }
#endif

#if defined(__AVX512VL__)
  static __forceinline vfloat8 shift_right_1(const vfloat8& x) {
    return align_shift_right<1>(zero,x);
  }
#else
  static __forceinline vfloat8 shift_right_1(const vfloat8& x) {
    const vfloat8 t0 = shuffle<1,2,3,0>(x);
    const vfloat8 t1 = shuffle4<1,0>(t0);
    return _mm256_blend_ps(t0,t1,0x88);
  }
#endif

  __forceinline vint8 floori(const vfloat8& a) {
    return vint8(floor(a));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Transpose
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline void transpose(const vfloat8& r0, const vfloat8& r1, const vfloat8& r2, const vfloat8& r3, vfloat8& c0, vfloat8& c1, vfloat8& c2, vfloat8& c3)
  {
    vfloat8 l02 = unpacklo(r0,r2);
    vfloat8 h02 = unpackhi(r0,r2);
    vfloat8 l13 = unpacklo(r1,r3);
    vfloat8 h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
    c3 = unpackhi(h02,h13);
  }

  __forceinline void transpose(const vfloat8& r0, const vfloat8& r1, const vfloat8& r2, const vfloat8& r3, vfloat8& c0, vfloat8& c1, vfloat8& c2)
  {
    vfloat8 l02 = unpacklo(r0,r2);
    vfloat8 h02 = unpackhi(r0,r2);
    vfloat8 l13 = unpacklo(r1,r3);
    vfloat8 h13 = unpackhi(r1,r3);
    c0 = unpacklo(l02,l13);
    c1 = unpackhi(l02,l13);
    c2 = unpacklo(h02,h13);
  }

  __forceinline void transpose(const vfloat8& r0, const vfloat8& r1, const vfloat8& r2, const vfloat8& r3, const vfloat8& r4, const vfloat8& r5, const vfloat8& r6, const vfloat8& r7,
                               vfloat8& c0, vfloat8& c1, vfloat8& c2, vfloat8& c3, vfloat8& c4, vfloat8& c5, vfloat8& c6, vfloat8& c7)
  {
    vfloat8 h0,h1,h2,h3; transpose(r0,r1,r2,r3,h0,h1,h2,h3);
    vfloat8 h4,h5,h6,h7; transpose(r4,r5,r6,r7,h4,h5,h6,h7);
    c0 = shuffle4<0,2>(h0,h4);
    c1 = shuffle4<0,2>(h1,h5);
    c2 = shuffle4<0,2>(h2,h6);
    c3 = shuffle4<0,2>(h3,h7);
    c4 = shuffle4<1,3>(h0,h4);
    c5 = shuffle4<1,3>(h1,h5);
    c6 = shuffle4<1,3>(h2,h6);
    c7 = shuffle4<1,3>(h3,h7);
  }

  __forceinline void transpose(const vfloat4& r0, const vfloat4& r1, const vfloat4& r2, const vfloat4& r3, const vfloat4& r4, const vfloat4& r5, const vfloat4& r6, const vfloat4& r7,
                               vfloat8& c0, vfloat8& c1, vfloat8& c2, vfloat8& c3)
  {
    transpose(vfloat8(r0,r4), vfloat8(r1,r5), vfloat8(r2,r6), vfloat8(r3,r7), c0, c1, c2, c3);
  }

  __forceinline void transpose(const vfloat4& r0, const vfloat4& r1, const vfloat4& r2, const vfloat4& r3, const vfloat4& r4, const vfloat4& r5, const vfloat4& r6, const vfloat4& r7,
                               vfloat8& c0, vfloat8& c1, vfloat8& c2)
  {
    transpose(vfloat8(r0,r4), vfloat8(r1,r5), vfloat8(r2,r6), vfloat8(r3,r7), c0, c1, c2);
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 vreduce_min2(const vfloat8& v) { return min(v,shuffle<1,0,3,2>(v)); }
  __forceinline vfloat8 vreduce_min4(const vfloat8& v) { vfloat8 v1 = vreduce_min2(v); return min(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline vfloat8 vreduce_min (const vfloat8& v) { vfloat8 v1 = vreduce_min4(v); return min(v1,shuffle4<1,0>(v1)); }

  __forceinline vfloat8 vreduce_max2(const vfloat8& v) { return max(v,shuffle<1,0,3,2>(v)); }
  __forceinline vfloat8 vreduce_max4(const vfloat8& v) { vfloat8 v1 = vreduce_max2(v); return max(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline vfloat8 vreduce_max (const vfloat8& v) { vfloat8 v1 = vreduce_max4(v); return max(v1,shuffle4<1,0>(v1)); }

  __forceinline vfloat8 vreduce_add2(const vfloat8& v) { return v + shuffle<1,0,3,2>(v); }
  __forceinline vfloat8 vreduce_add4(const vfloat8& v) { vfloat8 v1 = vreduce_add2(v); return v1 + shuffle<2,3,0,1>(v1); }
  __forceinline vfloat8 vreduce_add (const vfloat8& v) { vfloat8 v1 = vreduce_add4(v); return v1 + shuffle4<1,0>(v1); }

  __forceinline float reduce_min(const vfloat8& v) { return toScalar(vreduce_min(v)); }
  __forceinline float reduce_max(const vfloat8& v) { return toScalar(vreduce_max(v)); }
  __forceinline float reduce_add(const vfloat8& v) { return toScalar(vreduce_add(v)); }

  __forceinline size_t select_min(const vboolf8& valid, const vfloat8& v) 
  { 
    const vfloat8 a = select(valid,v,vfloat8(pos_inf)); 
    const vbool8 valid_min = valid & (a == vreduce_min(a));
    return bsf(movemask(any(valid_min) ? valid_min : valid)); 
  }

  __forceinline size_t select_max(const vboolf8& valid, const vfloat8& v) 
  { 
    const vfloat8 a = select(valid,v,vfloat8(neg_inf)); 
    const vbool8 valid_max = valid & (a == vreduce_max(a));
    return bsf(movemask(any(valid_max) ? valid_max : valid)); 
  }


  ////////////////////////////////////////////////////////////////////////////////
  /// Euclidian Space Operators (pairs of Vec3fa's)
  ////////////////////////////////////////////////////////////////////////////////

  //__forceinline vfloat8 dot(const vfloat8& a, const vfloat8& b) {
  //  return vreduce_add4(a*b);
  //}

  __forceinline vfloat8 dot(const vfloat8& a, const vfloat8& b) {
    return _mm256_dp_ps(a,b,0x7F);
  }

  __forceinline vfloat8 cross(const vfloat8& a, const vfloat8& b)
  {
    const vfloat8 a0 = a;
    const vfloat8 b0 = shuffle<1,2,0,3>(b);
    const vfloat8 a1 = shuffle<1,2,0,3>(a);
    const vfloat8 b1 = b;
    return shuffle<1,2,0,3>(msub(a0,b0,a1*b1));
  }

  //__forceinline float sqr_length (const vfloat<8>& a) { return dot(a,a); }
  //__forceinline float rcp_length (const vfloat<8>& a) { return rsqrt(dot(a,a)); }
  //__forceinline float rcp_length2(const vfloat<8>& a) { return rcp(dot(a,a)); }
  //__forceinline float length     (const vfloat<8>& a) { return sqrt(dot(a,a)); }
  __forceinline vfloat<8> normalize(const vfloat<8>& a) { return a*rsqrt(dot(a,a)); }
  //__forceinline float distance(const vfloat<8>& a, const vfloat<8>& b) { return length(a-b); }
  //__forceinline float halfArea(const vfloat<8>& d) { return madd(d.x,(d.y+d.z),d.y*d.z); }
  //__forceinline float area    (const vfloat<8>& d) { return 2.0f*halfArea(d); }
  //__forceinline vfloat<8> reflect(const vfloat<8>& V, const vfloat<8>& N) { return 2.0f*dot(V,N)*N-V; }

  //__forceinline vfloat<8> normalize_safe(const vfloat<8>& a) {
  //  const float d = dot(a,a); if (unlikely(d == 0.0f)) return a; else return a*rsqrt(d);
  //}

  ////////////////////////////////////////////////////////////////////////////////
  /// In Register Sorting
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vfloat8 sort_ascending(const vfloat8& v)
  {
    const vfloat8 a0 = v;
    const vfloat8 b0 = shuffle<1,0,3,2>(a0);
    const vfloat8 c0 = min(a0,b0);
    const vfloat8 d0 = max(a0,b0);
    const vfloat8 a1 = select<0x99 /* 0b10011001 */>(c0,d0);
    const vfloat8 b1 = shuffle<2,3,0,1>(a1);
    const vfloat8 c1 = min(a1,b1);
    const vfloat8 d1 = max(a1,b1);
    const vfloat8 a2 = select<0xc3 /* 0b11000011 */>(c1,d1);
    const vfloat8 b2 = shuffle<1,0,3,2>(a2);
    const vfloat8 c2 = min(a2,b2);
    const vfloat8 d2 = max(a2,b2);
    const vfloat8 a3 = select<0xa5 /* 0b10100101 */>(c2,d2);
    const vfloat8 b3 = shuffle4<1,0>(a3);
    const vfloat8 c3 = min(a3,b3);
    const vfloat8 d3 = max(a3,b3);
    const vfloat8 a4 = select<0xf /* 0b00001111 */>(c3,d3);
    const vfloat8 b4 = shuffle<2,3,0,1>(a4);
    const vfloat8 c4 = min(a4,b4);
    const vfloat8 d4 = max(a4,b4);
    const vfloat8 a5 = select<0x33 /* 0b00110011 */>(c4,d4);
    const vfloat8 b5 = shuffle<1,0,3,2>(a5);
    const vfloat8 c5 = min(a5,b5);
    const vfloat8 d5 = max(a5,b5);
    const vfloat8 a6 = select<0x55 /* 0b01010101 */>(c5,d5);
    return a6;
  }

   __forceinline vfloat8 sort_descending(const vfloat8& v)
  {
    const vfloat8 a0 = v;
    const vfloat8 b0 = shuffle<1,0,3,2>(a0);
    const vfloat8 c0 = max(a0,b0);
    const vfloat8 d0 = min(a0,b0);
    const vfloat8 a1 = select<0x99 /* 0b10011001 */>(c0,d0);
    const vfloat8 b1 = shuffle<2,3,0,1>(a1);
    const vfloat8 c1 = max(a1,b1);
    const vfloat8 d1 = min(a1,b1);
    const vfloat8 a2 = select<0xc3 /* 0b11000011 */>(c1,d1);
    const vfloat8 b2 = shuffle<1,0,3,2>(a2);
    const vfloat8 c2 = max(a2,b2);
    const vfloat8 d2 = min(a2,b2);
    const vfloat8 a3 = select<0xa5 /* 0b10100101 */>(c2,d2);
    const vfloat8 b3 = shuffle4<1,0>(a3);
    const vfloat8 c3 = max(a3,b3);
    const vfloat8 d3 = min(a3,b3);
    const vfloat8 a4 = select<0xf /* 0b00001111 */>(c3,d3);
    const vfloat8 b4 = shuffle<2,3,0,1>(a4);
    const vfloat8 c4 = max(a4,b4);
    const vfloat8 d4 = min(a4,b4);
    const vfloat8 a5 = select<0x33 /* 0b00110011 */>(c4,d4);
    const vfloat8 b5 = shuffle<1,0,3,2>(a5);
    const vfloat8 c5 = max(a5,b5);
    const vfloat8 d5 = min(a5,b5);
    const vfloat8 a6 = select<0x55 /* 0b01010101 */>(c5,d5);
    return a6;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator <<(embree_ostream cout, const vfloat8& a) {
    return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", " << a[4] << ", " << a[5] << ", " << a[6] << ", " << a[7] << ">";
  }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
