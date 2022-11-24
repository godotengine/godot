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
  /* 8-wide AVX integer type */
  template<>
  struct vint<8>
  {
    ALIGNED_STRUCT_(32);

    typedef vboolf8 Bool;
    typedef vint8   Int;
    typedef vfloat8 Float;

    enum  { size = 8 }; // number of SIMD elements
    union {             // data
      __m256i v;
      int i[8];
    }; 

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vint() {}
    __forceinline vint(const vint8& a) { v = a.v; }
    __forceinline vint8& operator =(const vint8& a) { v = a.v; return *this; }

    __forceinline vint(__m256i a) : v(a) {}
    __forceinline operator const __m256i&() const { return v; }
    __forceinline operator       __m256i&()       { return v; }

    __forceinline explicit vint(const vint4& a) : v(_mm256_insertf128_si256(_mm256_castsi128_si256(a),a,1)) {}
    __forceinline vint(const vint4& a, const vint4& b) : v(_mm256_insertf128_si256(_mm256_castsi128_si256(a),b,1)) {}
    __forceinline vint(const __m128i& a, const __m128i& b) : v(_mm256_insertf128_si256(_mm256_castsi128_si256(a),b,1)) {}
 
    __forceinline explicit vint(const int* a) : v(_mm256_castps_si256(_mm256_loadu_ps((const float*)a))) {}
    __forceinline vint(int a) : v(_mm256_set1_epi32(a)) {}
    __forceinline vint(int a, int b) : v(_mm256_set_epi32(b, a, b, a, b, a, b, a)) {}
    __forceinline vint(int a, int b, int c, int d) : v(_mm256_set_epi32(d, c, b, a, d, c, b, a)) {}
    __forceinline vint(int a, int b, int c, int d, int e, int f, int g, int h) : v(_mm256_set_epi32(h, g, f, e, d, c, b, a)) {}

    __forceinline explicit vint(__m256 a) : v(_mm256_cvtps_epi32(a)) {}

#if defined(__AVX512VL__)
    __forceinline explicit vint(const vboolf8& a) : v(_mm256_movm_epi32(a)) {}
#else
    __forceinline explicit vint(const vboolf8& a) : v(_mm256_castps_si256((__m256)a)) {}
#endif

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vint(ZeroTy)        : v(_mm256_setzero_si256()) {}
    __forceinline vint(OneTy)         : v(_mm256_set1_epi32(1)) {}
    __forceinline vint(PosInfTy)      : v(_mm256_set1_epi32(pos_inf)) {}
    __forceinline vint(NegInfTy)      : v(_mm256_set1_epi32(neg_inf)) {}
    __forceinline vint(StepTy)        : v(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)) {}
    __forceinline vint(ReverseStepTy) : v(_mm256_set_epi32(0, 1, 2, 3, 4, 5, 6, 7)) {}
    __forceinline vint(UndefinedTy)   : v(_mm256_undefined_si256()) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline vint8 load(const unsigned char* ptr)  { return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)ptr)); }
    static __forceinline vint8 loadu(const unsigned char* ptr) { return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)ptr)); }
    static __forceinline vint8 load(const unsigned short* ptr)  { return _mm256_cvtepu16_epi32(_mm_load_si128((__m128i*)ptr)); }
    static __forceinline vint8 loadu(const unsigned short* ptr) { return _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)ptr)); }

    static __forceinline vint8 load(const void* ptr) { return _mm256_load_si256((__m256i*)ptr); }
    static __forceinline vint8 loadu(const void* ptr) { return _mm256_loadu_si256((__m256i*)ptr); }

    static __forceinline void store (void* ptr, const vint8& v) { _mm256_store_si256((__m256i*)ptr,v); }
    static __forceinline void storeu(void* ptr, const vint8& v) { _mm256_storeu_ps((float*)ptr,_mm256_castsi256_ps(v)); }

#if defined(__AVX512VL__)

    static __forceinline vint8 compact(const vboolf8& mask, vint8 &v) {
      return _mm256_mask_compress_epi32(v, mask, v);
    }
    static __forceinline vint8 compact(const vboolf8& mask, vint8 &a, const vint8& b) {
      return _mm256_mask_compress_epi32(a, mask, b);
    }

    static __forceinline vint8 load (const vboolf8& mask, const void* ptr) { return _mm256_mask_load_epi32 (_mm256_setzero_si256(),mask,ptr); }
    static __forceinline vint8 loadu(const vboolf8& mask, const void* ptr) { return _mm256_mask_loadu_epi32(_mm256_setzero_si256(),mask,ptr); }

    static __forceinline void store (const vboolf8& mask, void* ptr, const vint8& v) { _mm256_mask_store_epi32 (ptr,mask,v); }
    static __forceinline void storeu(const vboolf8& mask, void* ptr, const vint8& v) { _mm256_mask_storeu_epi32(ptr,mask,v); }
#else
    static __forceinline vint8 load (const vboolf8& mask, const void* ptr) { return _mm256_castps_si256(_mm256_maskload_ps((float*)ptr,mask)); }
    static __forceinline vint8 loadu(const vboolf8& mask, const void* ptr) { return _mm256_castps_si256(_mm256_maskload_ps((float*)ptr,mask)); }

    static __forceinline void store (const vboolf8& mask, void* ptr, const vint8& v) { _mm256_maskstore_epi32((int*)ptr,mask,v); }
    static __forceinline void storeu(const vboolf8& mask, void* ptr, const vint8& v) { _mm256_maskstore_epi32((int*)ptr,mask,v); }
#endif
    
    static __forceinline vint8 load_nt(void* ptr) {
      return _mm256_stream_load_si256((__m256i*)ptr);
    }

    static __forceinline void store_nt(void* ptr, const vint8& v) {
      _mm256_stream_ps((float*)ptr,_mm256_castsi256_ps(v));
    }

    static __forceinline void store(unsigned char* ptr, const vint8& i)
    {
      for (size_t j=0; j<8; j++)
        ptr[j] = i[j];
    }

    static __forceinline void store(unsigned short* ptr, const vint8& v) {
      for (size_t i=0;i<8;i++)
        ptr[i] = (unsigned short)v[i];
    }

    template<int scale = 4>
    static __forceinline vint8 gather(const int *const ptr, const vint8& index) {
      return _mm256_i32gather_epi32(ptr, index, scale);
    }

    template<int scale = 4>
    static __forceinline vint8 gather(const vboolf8& mask, const int *const ptr, const vint8& index) {
      vint8 r = zero;
#if defined(__AVX512VL__)
      return _mm256_mmask_i32gather_epi32(r, mask, index, ptr, scale);
#else
      return _mm256_mask_i32gather_epi32(r, ptr, index, mask, scale);
#endif
    }

    template<int scale = 4>
    static __forceinline void scatter(void* ptr, const vint8& ofs, const vint8& v)
    {
#if defined(__AVX512VL__)
      _mm256_i32scatter_epi32((int*)ptr, ofs, v, scale);
#else
      *(int*)(((char*)ptr)+scale*ofs[0]) = v[0];
      *(int*)(((char*)ptr)+scale*ofs[1]) = v[1];
      *(int*)(((char*)ptr)+scale*ofs[2]) = v[2];
      *(int*)(((char*)ptr)+scale*ofs[3]) = v[3];
      *(int*)(((char*)ptr)+scale*ofs[4]) = v[4];
      *(int*)(((char*)ptr)+scale*ofs[5]) = v[5];
      *(int*)(((char*)ptr)+scale*ofs[6]) = v[6];
      *(int*)(((char*)ptr)+scale*ofs[7]) = v[7];
#endif
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf8& mask, void* ptr, const vint8& ofs, const vint8& v)
    {
#if defined(__AVX512VL__)
      _mm256_mask_i32scatter_epi32((int*)ptr, mask, ofs, v, scale);
#else
      if (likely(mask[0])) *(int*)(((char*)ptr)+scale*ofs[0]) = v[0];
      if (likely(mask[1])) *(int*)(((char*)ptr)+scale*ofs[1]) = v[1];
      if (likely(mask[2])) *(int*)(((char*)ptr)+scale*ofs[2]) = v[2];
      if (likely(mask[3])) *(int*)(((char*)ptr)+scale*ofs[3]) = v[3];
      if (likely(mask[4])) *(int*)(((char*)ptr)+scale*ofs[4]) = v[4];
      if (likely(mask[5])) *(int*)(((char*)ptr)+scale*ofs[5]) = v[5];
      if (likely(mask[6])) *(int*)(((char*)ptr)+scale*ofs[6]) = v[6];
      if (likely(mask[7])) *(int*)(((char*)ptr)+scale*ofs[7]) = v[7];
#endif
    }

    static __forceinline vint8 broadcast64(const long long &a) { return _mm256_set1_epi64x(a); }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const int& operator [](size_t index) const { assert(index < 8); return i[index]; }
    __forceinline       int& operator [](size_t index)       { assert(index < 8); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  static __forceinline vboolf8 asBool(const vint8& a) { return _mm256_movepi32_mask(a); }
#else
  static __forceinline vboolf8 asBool(const vint8& a) { return _mm256_castsi256_ps(a); }
#endif

  __forceinline vint8 operator +(const vint8& a) { return a; }
  __forceinline vint8 operator -(const vint8& a) { return _mm256_sub_epi32(_mm256_setzero_si256(), a); }
  __forceinline vint8 abs       (const vint8& a) { return _mm256_abs_epi32(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint8 operator +(const vint8& a, const vint8& b) { return _mm256_add_epi32(a, b); }
  __forceinline vint8 operator +(const vint8& a, int          b) { return a + vint8(b); }
  __forceinline vint8 operator +(int          a, const vint8& b) { return vint8(a) + b; }

  __forceinline vint8 operator -(const vint8& a, const vint8& b) { return _mm256_sub_epi32(a, b); }
  __forceinline vint8 operator -(const vint8& a, int          b) { return a - vint8(b); }
  __forceinline vint8 operator -(int          a, const vint8& b) { return vint8(a) - b; }

  __forceinline vint8 operator *(const vint8& a, const vint8& b) { return _mm256_mullo_epi32(a, b); }
  __forceinline vint8 operator *(const vint8& a, int          b) { return a * vint8(b); }
  __forceinline vint8 operator *(int          a, const vint8& b) { return vint8(a) * b; }

  __forceinline vint8 operator &(const vint8& a, const vint8& b) { return _mm256_and_si256(a, b); }
  __forceinline vint8 operator &(const vint8& a, int          b) { return a & vint8(b); }
  __forceinline vint8 operator &(int          a, const vint8& b) { return vint8(a) & b; }

  __forceinline vint8 operator |(const vint8& a, const vint8& b) { return _mm256_or_si256(a, b); }
  __forceinline vint8 operator |(const vint8& a, int          b) { return a | vint8(b); }
  __forceinline vint8 operator |(int          a, const vint8& b) { return vint8(a) | b; }

  __forceinline vint8 operator ^(const vint8& a, const vint8& b) { return _mm256_xor_si256(a, b); }
  __forceinline vint8 operator ^(const vint8& a, int          b) { return a ^ vint8(b); }
  __forceinline vint8 operator ^(int          a, const vint8& b) { return vint8(a) ^ b; }

  __forceinline vint8 operator <<(const vint8& a, int n) { return _mm256_slli_epi32(a, n); }
  __forceinline vint8 operator >>(const vint8& a, int n) { return _mm256_srai_epi32(a, n); }

  __forceinline vint8 operator <<(const vint8& a, const vint8& n) { return _mm256_sllv_epi32(a, n); }
  __forceinline vint8 operator >>(const vint8& a, const vint8& n) { return _mm256_srav_epi32(a, n); }

  __forceinline vint8 sll(const vint8& a, int b) { return _mm256_slli_epi32(a, b); }
  __forceinline vint8 sra(const vint8& a, int b) { return _mm256_srai_epi32(a, b); }
  __forceinline vint8 srl(const vint8& a, int b) { return _mm256_srli_epi32(a, b); }

  __forceinline vint8 sll(const vint8& a, const vint8& b) { return _mm256_sllv_epi32(a, b); }
  __forceinline vint8 sra(const vint8& a, const vint8& b) { return _mm256_srav_epi32(a, b); }
  __forceinline vint8 srl(const vint8& a, const vint8& b) { return _mm256_srlv_epi32(a, b); }
  
  __forceinline vint8 min(const vint8& a, const vint8& b) { return _mm256_min_epi32(a, b); }
  __forceinline vint8 min(const vint8& a, int          b) { return min(a,vint8(b)); }
  __forceinline vint8 min(int          a, const vint8& b) { return min(vint8(a),b); }

  __forceinline vint8 max(const vint8& a, const vint8& b) { return _mm256_max_epi32(a, b); }
  __forceinline vint8 max(const vint8& a, int          b) { return max(a,vint8(b)); }
  __forceinline vint8 max(int          a, const vint8& b) { return max(vint8(a),b); }

  __forceinline vint8 umin(const vint8& a, const vint8& b) { return _mm256_min_epu32(a, b); }
  __forceinline vint8 umax(const vint8& a, const vint8& b) { return _mm256_max_epu32(a, b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint8& operator +=(vint8& a, const vint8& b) { return a = a + b; }
  __forceinline vint8& operator +=(vint8& a, int          b) { return a = a + b; }
  
  __forceinline vint8& operator -=(vint8& a, const vint8& b) { return a = a - b; }
  __forceinline vint8& operator -=(vint8& a, int          b) { return a = a - b; }
  
  __forceinline vint8& operator *=(vint8& a, const vint8& b) { return a = a * b; }
  __forceinline vint8& operator *=(vint8& a, int          b) { return a = a * b; }
  
  __forceinline vint8& operator &=(vint8& a, const vint8& b) { return a = a & b; }
  __forceinline vint8& operator &=(vint8& a, int          b) { return a = a & b; }
  
  __forceinline vint8& operator |=(vint8& a, const vint8& b) { return a = a | b; }
  __forceinline vint8& operator |=(vint8& a, int          b) { return a = a | b; }
  
  __forceinline vint8& operator <<=(vint8& a, const int b) { return a = a << b; }
  __forceinline vint8& operator >>=(vint8& a, const int b) { return a = a >> b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  static __forceinline vboolf8 operator ==(const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_EQ); }
  static __forceinline vboolf8 operator !=(const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_NE); }
  static __forceinline vboolf8 operator < (const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_LT); }
  static __forceinline vboolf8 operator >=(const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_GE); }
  static __forceinline vboolf8 operator > (const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_GT); }
  static __forceinline vboolf8 operator <=(const vint8& a, const vint8& b) { return _mm256_cmp_epi32_mask(a,b,_MM_CMPINT_LE); }

  static __forceinline vint8 select(const vboolf8& m, const vint8& t, const vint8& f) {
    return _mm256_mask_blend_epi32(m, (__m256i)f, (__m256i)t);
  }
#else
  static __forceinline vboolf8 operator ==(const vint8& a, const vint8& b) { return _mm256_castsi256_ps(_mm256_cmpeq_epi32(a, b)); }
  static __forceinline vboolf8 operator !=(const vint8& a, const vint8& b) { return !(a == b); }
  static __forceinline vboolf8 operator < (const vint8& a, const vint8& b) { return _mm256_castsi256_ps(_mm256_cmpgt_epi32(b, a)); }
  static __forceinline vboolf8 operator >=(const vint8& a, const vint8& b) { return !(a <  b); }
  static __forceinline vboolf8 operator > (const vint8& a, const vint8& b) { return _mm256_castsi256_ps(_mm256_cmpgt_epi32(a, b)); }
  static __forceinline vboolf8 operator <=(const vint8& a, const vint8& b) { return !(a >  b); }

  static __forceinline vint8 select(const vboolf8& m, const vint8& t, const vint8& f) {
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(f), _mm256_castsi256_ps(t), m));
  }
#endif

  template<int mask>
  __forceinline vint8 select(const vint8& t, const vint8& f) {
    return _mm256_blend_epi32(f, t, mask);
  }

  __forceinline vboolf8 operator ==(const vint8& a, int          b) { return a == vint8(b); }
  __forceinline vboolf8 operator ==(int          a, const vint8& b) { return vint8(a) == b; }

  __forceinline vboolf8 operator !=(const vint8& a, int          b) { return a != vint8(b); }
  __forceinline vboolf8 operator !=(int          a, const vint8& b) { return vint8(a) != b; }

  __forceinline vboolf8 operator < (const vint8& a, int          b) { return a <  vint8(b); }
  __forceinline vboolf8 operator < (int          a, const vint8& b) { return vint8(a) <  b; }

  __forceinline vboolf8 operator >=(const vint8& a, int          b) { return a >= vint8(b); }
  __forceinline vboolf8 operator >=(int          a, const vint8& b) { return vint8(a) >= b; }

  __forceinline vboolf8 operator > (const vint8& a, int          b) { return a >  vint8(b); }
  __forceinline vboolf8 operator > (int          a, const vint8& b) { return vint8(a) >  b; }

  __forceinline vboolf8 operator <=(const vint8& a, int          b) { return a <= vint8(b); }
  __forceinline vboolf8 operator <=(int          a, const vint8& b) { return vint8(a) <= b; }

  __forceinline vboolf8 eq(const vint8& a, const vint8& b) { return a == b; }
  __forceinline vboolf8 ne(const vint8& a, const vint8& b) { return a != b; }
  __forceinline vboolf8 lt(const vint8& a, const vint8& b) { return a <  b; }
  __forceinline vboolf8 ge(const vint8& a, const vint8& b) { return a >= b; }
  __forceinline vboolf8 gt(const vint8& a, const vint8& b) { return a >  b; }
  __forceinline vboolf8 le(const vint8& a, const vint8& b) { return a <= b; }

#if defined(__AVX512VL__)
  static __forceinline vboolf8 eq(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_EQ); }
  static __forceinline vboolf8 ne(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_NE); }
  static __forceinline vboolf8 lt(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_LT); }
  static __forceinline vboolf8 ge(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_GE); }
  static __forceinline vboolf8 gt(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_GT); }
  static __forceinline vboolf8 le(const vboolf8& mask, const vint8& a, const vint8& b) { return _mm256_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  static __forceinline vboolf8 eq(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a == b); }
  static __forceinline vboolf8 ne(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a != b); }
  static __forceinline vboolf8 lt(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a <  b); }
  static __forceinline vboolf8 ge(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a >= b); }
  static __forceinline vboolf8 gt(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a >  b); }
  static __forceinline vboolf8 le(const vboolf8& mask, const vint8& a, const vint8& b) { return mask & (a <= b); }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint8 unpacklo(const vint8& a, const vint8& b) { return _mm256_unpacklo_epi32(a, b); }
  __forceinline vint8 unpackhi(const vint8& a, const vint8& b) { return _mm256_unpackhi_epi32(a, b); }

  template<int i>
  __forceinline vint8 shuffle(const vint8& v) {
    return _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(v), _MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1>
  __forceinline vint8 shuffle4(const vint8& v) {
    return _mm256_permute2f128_si256(v, v, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1>
  __forceinline vint8 shuffle4(const vint8& a, const vint8& b) {
    return _mm256_permute2f128_si256(a, b, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vint8 shuffle(const vint8& v) {
    return _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vint8 shuffle(const vint8& a, const vint8& b) {
    return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<> __forceinline vint8 shuffle<0, 0, 2, 2>(const vint8& v) { return _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(v))); }
  template<> __forceinline vint8 shuffle<1, 1, 3, 3>(const vint8& v) { return _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(v))); }
  template<> __forceinline vint8 shuffle<0, 1, 0, 1>(const vint8& v) { return _mm256_castps_si256(_mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(_mm256_castsi256_ps(v))))); }

  __forceinline vint8 broadcast(const int* ptr) { return _mm256_castps_si256(_mm256_broadcast_ss((const float*)ptr)); }

  template<int i> __forceinline vint8 insert4(const vint8& a, const vint4& b) { return _mm256_insertf128_si256(a, b, i); }
  template<int i> __forceinline vint4 extract4(const vint8& a) { return _mm256_extractf128_si256(a, i); }
  template<> __forceinline vint4 extract4<0>(const vint8& a) { return _mm256_castsi256_si128(a); }

  __forceinline int toScalar(const vint8& v) { return _mm_cvtsi128_si32(_mm256_castsi256_si128(v)); }

#if !defined(__aarch64__)
  __forceinline vint8 permute(const vint8& v, const __m256i& index) {
    return _mm256_permutevar8x32_epi32(v, index);
  }

  __forceinline vint8 shuffle(const vint8& v, const __m256i& index) {
    return _mm256_castps_si256(_mm256_permutevar_ps(_mm256_castsi256_ps(v), index));
  }

  template<int i>
  static __forceinline vint8 align_shift_right(const vint8& a, const vint8& b) {
#if defined(__AVX512VL__)
    return _mm256_alignr_epi32(a, b, i);    
#else
    return _mm256_alignr_epi8(a, b, 4*i);
#endif
  }  

#endif


  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint8 vreduce_min2(const vint8& v) { return min(v,shuffle<1,0,3,2>(v)); }
  __forceinline vint8 vreduce_min4(const vint8& v) { vint8 v1 = vreduce_min2(v); return min(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline vint8 vreduce_min (const vint8& v) { vint8 v1 = vreduce_min4(v); return min(v1,shuffle4<1,0>(v1)); }

  __forceinline vint8 vreduce_max2(const vint8& v) { return max(v,shuffle<1,0,3,2>(v)); }
  __forceinline vint8 vreduce_max4(const vint8& v) { vint8 v1 = vreduce_max2(v); return max(v1,shuffle<2,3,0,1>(v1)); }
  __forceinline vint8 vreduce_max (const vint8& v) { vint8 v1 = vreduce_max4(v); return max(v1,shuffle4<1,0>(v1)); }

  __forceinline vint8 vreduce_add2(const vint8& v) { return v + shuffle<1,0,3,2>(v); }
  __forceinline vint8 vreduce_add4(const vint8& v) { vint8 v1 = vreduce_add2(v); return v1 + shuffle<2,3,0,1>(v1); }
  __forceinline vint8 vreduce_add (const vint8& v) { vint8 v1 = vreduce_add4(v); return v1 + shuffle4<1,0>(v1); }

  __forceinline int reduce_min(const vint8& v) { return toScalar(vreduce_min(v)); }
  __forceinline int reduce_max(const vint8& v) { return toScalar(vreduce_max(v)); }
  __forceinline int reduce_add(const vint8& v) { return toScalar(vreduce_add(v)); }

  __forceinline size_t select_min(const vint8& v) { return bsf(movemask(v == vreduce_min(v))); }
  __forceinline size_t select_max(const vint8& v) { return bsf(movemask(v == vreduce_max(v))); }

  __forceinline size_t select_min(const vboolf8& valid, const vint8& v) { const vint8 a = select(valid,v,vint8(pos_inf)); return bsf(movemask(valid & (a == vreduce_min(a)))); }
  __forceinline size_t select_max(const vboolf8& valid, const vint8& v) { const vint8 a = select(valid,v,vint8(neg_inf)); return bsf(movemask(valid & (a == vreduce_max(a)))); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Sorting networks
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint8 usort_ascending(const vint8& v)
  {
    const vint8 a0 = v;
    const vint8 b0 = shuffle<1,0,3,2>(a0);
    const vint8 c0 = umin(a0,b0);
    const vint8 d0 = umax(a0,b0);
    const vint8 a1 = select<0x99 /* 0b10011001 */>(c0,d0);
    const vint8 b1 = shuffle<2,3,0,1>(a1);
    const vint8 c1 = umin(a1,b1);
    const vint8 d1 = umax(a1,b1);
    const vint8 a2 = select<0xc3 /* 0b11000011 */>(c1,d1);
    const vint8 b2 = shuffle<1,0,3,2>(a2);
    const vint8 c2 = umin(a2,b2);
    const vint8 d2 = umax(a2,b2);
    const vint8 a3 = select<0xa5 /* 0b10100101 */>(c2,d2);
    const vint8 b3 = shuffle4<1,0>(a3);
    const vint8 c3 = umin(a3,b3);
    const vint8 d3 = umax(a3,b3);
    const vint8 a4 = select<0xf /* 0b00001111 */>(c3,d3);
    const vint8 b4 = shuffle<2,3,0,1>(a4);
    const vint8 c4 = umin(a4,b4);
    const vint8 d4 = umax(a4,b4);
    const vint8 a5 = select<0x33 /* 0b00110011 */>(c4,d4);
    const vint8 b5 = shuffle<1,0,3,2>(a5);
    const vint8 c5 = umin(a5,b5);
    const vint8 d5 = umax(a5,b5);
    const vint8 a6 = select<0x55 /* 0b01010101 */>(c5,d5);
    return a6;
  }

  __forceinline vint8 usort_descending(const vint8& v)
  {
    const vint8 a0 = v;
    const vint8 b0 = shuffle<1,0,3,2>(a0);
    const vint8 c0 = umax(a0,b0);
    const vint8 d0 = umin(a0,b0);
    const vint8 a1 = select<0x99 /* 0b10011001 */>(c0,d0);
    const vint8 b1 = shuffle<2,3,0,1>(a1);
    const vint8 c1 = umax(a1,b1);
    const vint8 d1 = umin(a1,b1);
    const vint8 a2 = select<0xc3 /* 0b11000011 */>(c1,d1);
    const vint8 b2 = shuffle<1,0,3,2>(a2);
    const vint8 c2 = umax(a2,b2);
    const vint8 d2 = umin(a2,b2);
    const vint8 a3 = select<0xa5 /* 0b10100101 */>(c2,d2);
    const vint8 b3 = shuffle4<1,0>(a3);
    const vint8 c3 = umax(a3,b3);
    const vint8 d3 = umin(a3,b3);
    const vint8 a4 = select<0xf /* 0b00001111 */>(c3,d3);
    const vint8 b4 = shuffle<2,3,0,1>(a4);
    const vint8 c4 = umax(a4,b4);
    const vint8 d4 = umin(a4,b4);
    const vint8 a5 = select<0x33 /* 0b00110011 */>(c4,d4);
    const vint8 b5 = shuffle<1,0,3,2>(a5);
    const vint8 c5 = umax(a5,b5);
    const vint8 d5 = umin(a5,b5);
    const vint8 a6 = select<0x55 /* 0b01010101 */>(c5,d5);
    return a6;
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator <<(embree_ostream cout, const vint8& a) {
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
