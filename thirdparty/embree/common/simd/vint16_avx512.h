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
  /* 16-wide AVX-512 integer type */
  template<>
  struct vint<16>
  {
    ALIGNED_STRUCT_(64);
    
    typedef vboolf16 Bool;
    typedef vint16   Int;
    typedef vfloat16 Float;

    enum  { size = 16 }; // number of SIMD elements
    union {              // data
      __m512i v; 
      int i[16]; 
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
       
    __forceinline vint() {}
    __forceinline vint(const vint16& t) { v = t.v; }
    __forceinline vint16& operator =(const vint16& f) { v = f.v; return *this; }

    __forceinline vint(const __m512i& t) { v = t; }
    __forceinline operator __m512i() const { return v; }
    __forceinline operator __m256i() const { return _mm512_castsi512_si256(v); }

    __forceinline vint(int i) {
      v = _mm512_set1_epi32(i);
    }
    
    __forceinline vint(int a, int b, int c, int d) {
      v = _mm512_set4_epi32(d,c,b,a);      
    }

    __forceinline vint(int a0 , int a1 , int a2 , int a3,
                       int a4 , int a5 , int a6 , int a7,
                       int a8 , int a9 , int a10, int a11,
                       int a12, int a13, int a14, int a15)
    {
      v = _mm512_set_epi32(a15,a14,a13,a12,a11,a10,a9,a8,a7,a6,a5,a4,a3,a2,a1,a0);
    }

    __forceinline vint(const vint4& i) {
      v = _mm512_broadcast_i32x4(i);
    }

    __forceinline vint(const vint4& a, const vint4& b, const vint4& c, const vint4& d) {
      v = _mm512_castsi128_si512(a);
      v = _mm512_inserti32x4(v, b, 1);
      v = _mm512_inserti32x4(v, c, 2);
      v = _mm512_inserti32x4(v, d, 3);
    }

    __forceinline vint(const vint8& i) {
      v = _mm512_castps_si512(_mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castsi256_pd(i))));
    }

    __forceinline vint(const vint8& a, const vint8& b) {
      v = _mm512_castsi256_si512(a);
      v = _mm512_inserti64x4(v, b, 1);
    }
   
    __forceinline explicit vint(const __m512& f) {
      v = _mm512_cvtps_epi32(f);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vint(ZeroTy)   : v(_mm512_setzero_epi32()) {}
    __forceinline vint(OneTy)    : v(_mm512_set1_epi32(1)) {}
    __forceinline vint(PosInfTy) : v(_mm512_set1_epi32(pos_inf)) {}
    __forceinline vint(NegInfTy) : v(_mm512_set1_epi32(neg_inf)) {}
    __forceinline vint(StepTy)   : v(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)) {}
    __forceinline vint(ReverseStepTy) : v(_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline vint16 load (const void* addr) { return _mm512_load_si512((int*)addr); }

    static __forceinline vint16 load(const unsigned char* ptr) { return _mm512_cvtepu8_epi32(_mm_load_si128((__m128i*)ptr)); }
    static __forceinline vint16 load(const unsigned short* ptr) { return _mm512_cvtepu16_epi32(_mm256_load_si256((__m256i*)ptr)); }

    static __forceinline vint16 loadu(const unsigned char* ptr) { return _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ptr)); }
    static __forceinline vint16 loadu(const unsigned short* ptr) { return _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)ptr)); }

    static __forceinline vint16 loadu(const void* addr) { return _mm512_loadu_si512(addr); }

    static __forceinline vint16 load (const vboolf16& mask, const void* addr) { return _mm512_mask_load_epi32 (_mm512_setzero_epi32(),mask,addr); }
    static __forceinline vint16 loadu(const vboolf16& mask, const void* addr) { return _mm512_mask_loadu_epi32(_mm512_setzero_epi32(),mask,addr); }

    static __forceinline void store (void* ptr, const vint16& v) { _mm512_store_si512 (ptr,v); }
    static __forceinline void storeu(void* ptr, const vint16& v) { _mm512_storeu_si512(ptr,v); }
 
    static __forceinline void store (const vboolf16& mask, void* addr, const vint16& v2) { _mm512_mask_store_epi32(addr,mask,v2); }
    static __forceinline void storeu(const vboolf16& mask, void* ptr,  const vint16& f) { _mm512_mask_storeu_epi32((int*)ptr,mask,f); }

    static __forceinline void store_nt(void* __restrict__ ptr, const vint16& a) { _mm512_stream_si512((__m512i*)ptr,a); }

    static __forceinline vint16 compact(const vboolf16& mask, vint16 &v) {
      return _mm512_mask_compress_epi32(v,mask,v);
    }

    static __forceinline vint16 compact(const vboolf16& mask, const vint16 &a, vint16 &b) {
      return _mm512_mask_compress_epi32(a,mask,b);
    }

    static __forceinline vint16 expand(const vboolf16& mask, const vint16& a, vint16& b) {
      return _mm512_mask_expand_epi32(b,mask,a);
    }

    template<int scale = 4>
    static __forceinline vint16 gather(const int* ptr, const vint16& index) {
      return _mm512_i32gather_epi32(index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline vint16 gather(const vboolf16& mask, const int* ptr, const vint16& index) {
      return _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(),mask,index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline vint16 gather(const vboolf16& mask, vint16& dest, const int* ptr, const vint16& index) {
      return _mm512_mask_i32gather_epi32(dest,mask,index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline void scatter(int* ptr, const vint16& index, const vint16& v) {
      _mm512_i32scatter_epi32((int*)ptr,index,v,scale);
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf16& mask, int* ptr, const vint16& index, const vint16& v) {
      _mm512_mask_i32scatter_epi32((int*)ptr,mask,index,v,scale);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline       int& operator [](size_t index)       { assert(index < 16); return i[index]; }
    __forceinline const int& operator [](size_t index) const { assert(index < 16); return i[index]; }

    __forceinline unsigned int uint    (size_t index) const { assert(index < 16); return ((unsigned int*)i)[index]; }
    __forceinline size_t&      uint64_t(size_t index) const { assert(index < 8);  return ((size_t*)i)[index]; }
  };
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf16 asBool(const vint16& a) { return _mm512_movepi32_mask(a); }

  __forceinline vint16 operator +(const vint16& a) { return a; }
  __forceinline vint16 operator -(const vint16& a) { return _mm512_sub_epi32(_mm512_setzero_epi32(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint16 operator +(const vint16& a, const vint16& b) { return _mm512_add_epi32(a, b); }
  __forceinline vint16 operator +(const vint16& a, int           b) { return a + vint16(b); }
  __forceinline vint16 operator +(int           a, const vint16& b) { return vint16(a) + b; }

  __forceinline vint16 operator -(const vint16& a, const vint16& b) { return _mm512_sub_epi32(a, b); }
  __forceinline vint16 operator -(const vint16& a, int           b) { return a - vint16(b); }
  __forceinline vint16 operator -(int           a, const vint16& b) { return vint16(a) - b; }

  __forceinline vint16 operator *(const vint16& a, const vint16& b) { return _mm512_mullo_epi32(a, b); }
  __forceinline vint16 operator *(const vint16& a, int           b) { return a * vint16(b); }
  __forceinline vint16 operator *(int           a, const vint16& b) { return vint16(a) * b; }

  __forceinline vint16 operator &(const vint16& a, const vint16& b) { return _mm512_and_epi32(a, b); }
  __forceinline vint16 operator &(const vint16& a, int           b) { return a & vint16(b); }
  __forceinline vint16 operator &(int           a, const vint16& b) { return vint16(a) & b; }

  __forceinline vint16 operator |(const vint16& a, const vint16& b) { return _mm512_or_epi32(a, b); }
  __forceinline vint16 operator |(const vint16& a, int           b) { return a | vint16(b); }
  __forceinline vint16 operator |(int           a, const vint16& b) { return vint16(a) | b; }

  __forceinline vint16 operator ^(const vint16& a, const vint16& b) { return _mm512_xor_epi32(a, b); }
  __forceinline vint16 operator ^(const vint16& a, int           b) { return a ^ vint16(b); }
  __forceinline vint16 operator ^(int           a, const vint16& b) { return vint16(a) ^ b; }

  __forceinline vint16 operator <<(const vint16& a, int n) { return _mm512_slli_epi32(a, n); }
  __forceinline vint16 operator >>(const vint16& a, int n) { return _mm512_srai_epi32(a, n); }

  __forceinline vint16 operator <<(const vint16& a, const vint16& n) { return _mm512_sllv_epi32(a, n); }
  __forceinline vint16 operator >>(const vint16& a, const vint16& n) { return _mm512_srav_epi32(a, n); }

  __forceinline vint16 sll (const vint16& a, int b) { return _mm512_slli_epi32(a, b); }
  __forceinline vint16 sra (const vint16& a, int b) { return _mm512_srai_epi32(a, b); }
  __forceinline vint16 srl (const vint16& a, int b) { return _mm512_srli_epi32(a, b); }
  
  __forceinline vint16 min(const vint16& a, const vint16& b) { return _mm512_min_epi32(a, b); }
  __forceinline vint16 min(const vint16& a, int           b) { return min(a,vint16(b)); }
  __forceinline vint16 min(int           a, const vint16& b) { return min(vint16(a),b); }

  __forceinline vint16 max(const vint16& a, const vint16& b) { return _mm512_max_epi32(a, b); }
  __forceinline vint16 max(const vint16& a, int           b) { return max(a,vint16(b)); }
  __forceinline vint16 max(int           a, const vint16& b) { return max(vint16(a),b); }
  
  __forceinline vint16 umin(const vint16& a, const vint16& b) { return _mm512_min_epu32(a, b); }
  __forceinline vint16 umax(const vint16& a, const vint16& b) { return _mm512_max_epu32(a, b); }

  __forceinline vint16 mask_add(const vboolf16& mask, vint16& c, const vint16& a, const vint16& b) { return _mm512_mask_add_epi32(c,mask,a,b); }
  __forceinline vint16 mask_sub(const vboolf16& mask, vint16& c, const vint16& a, const vint16& b) { return _mm512_mask_sub_epi32(c,mask,a,b); }

  __forceinline vint16 mask_and(const vboolf16& m, vint16& c, const vint16& a, const vint16& b) { return _mm512_mask_and_epi32(c,m,a,b); }
  __forceinline vint16 mask_or (const vboolf16& m, vint16& c, const vint16& a, const vint16& b) { return _mm512_mask_or_epi32(c,m,a,b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint16& operator +=(vint16& a, const vint16& b) { return a = a + b; }
  __forceinline vint16& operator +=(vint16& a, int           b) { return a = a + b; }
  
  __forceinline vint16& operator -=(vint16& a, const vint16& b) { return a = a - b; }
  __forceinline vint16& operator -=(vint16& a, int           b) { return a = a - b; }

  __forceinline vint16& operator *=(vint16& a, const vint16& b) { return a = a * b; }
  __forceinline vint16& operator *=(vint16& a, int           b) { return a = a * b; }
  
  __forceinline vint16& operator &=(vint16& a, const vint16& b) { return a = a & b; }
  __forceinline vint16& operator &=(vint16& a, int           b) { return a = a & b; }
  
  __forceinline vint16& operator |=(vint16& a, const vint16& b) { return a = a | b; }
  __forceinline vint16& operator |=(vint16& a, int           b) { return a = a | b; }
  
  __forceinline vint16& operator <<=(vint16& a, int b) { return a = a << b; }
  __forceinline vint16& operator >>=(vint16& a, int b) { return a = a >> b; }


  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf16 operator ==(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 operator ==(const vint16& a, int           b) { return a == vint16(b); }
  __forceinline vboolf16 operator ==(int           a, const vint16& b) { return vint16(a) == b; }
  
  __forceinline vboolf16 operator !=(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 operator !=(const vint16& a, int           b) { return a != vint16(b); }
  __forceinline vboolf16 operator !=(int           a, const vint16& b) { return vint16(a) != b; }
  
  __forceinline vboolf16 operator < (const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 operator < (const vint16& a, int           b) { return a <  vint16(b); }
  __forceinline vboolf16 operator < (int           a, const vint16& b) { return vint16(a) <  b; }
  
  __forceinline vboolf16 operator >=(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 operator >=(const vint16& a, int           b) { return a >= vint16(b); }
  __forceinline vboolf16 operator >=(int           a, const vint16& b) { return vint16(a) >= b; }

  __forceinline vboolf16 operator > (const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 operator > (const vint16& a, int           b) { return a >  vint16(b); }
  __forceinline vboolf16 operator > (int           a, const vint16& b) { return vint16(a) >  b; }

  __forceinline vboolf16 operator <=(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 operator <=(const vint16& a, int           b) { return a <= vint16(b); }
  __forceinline vboolf16 operator <=(int           a, const vint16& b) { return vint16(a) <= b; }

  __forceinline vboolf16 eq(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 ne(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 lt(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 ge(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 gt(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 le(const vint16& a, const vint16& b) { return _mm512_cmp_epi32_mask(a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 uint_le(const vint16& a, const vint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 uint_gt(const vint16& a, const vint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_GT); }

  __forceinline vboolf16 eq(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 ne(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 lt(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 ge(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 gt(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 le(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epi32_mask(mask,a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 uint_le(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 uint_gt(const vboolf16 mask, const vint16& a, const vint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_GT); }
    
 
  __forceinline vint16 select(const vboolf16& m, const vint16& t, const vint16& f) {
    return _mm512_mask_or_epi32(f,m,t,t); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint16 unpacklo(const vint16& a, const vint16& b) { return _mm512_unpacklo_epi32(a, b); }
  __forceinline vint16 unpackhi(const vint16& a, const vint16& b) { return _mm512_unpackhi_epi32(a, b); }

  template<int i>
    __forceinline vint16 shuffle(const vint16& v) {
    return _mm512_castps_si512(_mm512_permute_ps(_mm512_castsi512_ps(v), _MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vint16 shuffle(const vint16& v) {
    return _mm512_castps_si512(_mm512_permute_ps(_mm512_castsi512_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i>
  __forceinline vint16 shuffle4(const vint16& v) {
    return _mm512_castps_si512(_mm512_shuffle_f32x4(_mm512_castsi512_ps(v), _mm512_castsi512_ps(v), _MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vint16 shuffle4(const vint16& v) {
    return _mm512_castps_si512(_mm512_shuffle_f32x4(_mm512_castsi512_ps(v), _mm512_castsi512_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i>
  __forceinline vint16 align_shift_right(const vint16& a, const vint16& b) {
    return _mm512_alignr_epi32(a, b, i);
  };

  __forceinline int toScalar(const vint16& v) {
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(v));
  }

  template<int i> __forceinline vint16 insert4(const vint16& a, const vint4& b) { return _mm512_inserti32x4(a, b, i); }

  template<int N, int i>
  vint<N> extractN(const vint16& v);

  template<> __forceinline vint4 extractN<4,0>(const vint16& v) { return _mm512_castsi512_si128(v);       }
  template<> __forceinline vint4 extractN<4,1>(const vint16& v) { return _mm512_extracti32x4_epi32(v, 1); }
  template<> __forceinline vint4 extractN<4,2>(const vint16& v) { return _mm512_extracti32x4_epi32(v, 2); }
  template<> __forceinline vint4 extractN<4,3>(const vint16& v) { return _mm512_extracti32x4_epi32(v, 3); }

  template<> __forceinline vint8 extractN<8,0>(const vint16& v) { return _mm512_castsi512_si256(v);       }
  template<> __forceinline vint8 extractN<8,1>(const vint16& v) { return _mm512_extracti32x8_epi32(v, 1); }

  template<int i> __forceinline vint4 extract4   (const vint16& v) { return _mm512_extracti32x4_epi32(v, i); }
  template<>      __forceinline vint4 extract4<0>(const vint16& v) { return _mm512_castsi512_si128(v);       }

  template<int i> __forceinline vint8 extract8   (const vint16& v) { return _mm512_extracti32x8_epi32(v, i); }
  template<>      __forceinline vint8 extract8<0>(const vint16& v) { return _mm512_castsi512_si256(v);       }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint16 vreduce_min2(vint16 x) {                      return min(x, shuffle<1,0,3,2>(x)); }
  __forceinline vint16 vreduce_min4(vint16 x) { x = vreduce_min2(x); return min(x, shuffle<2,3,0,1>(x)); }
  __forceinline vint16 vreduce_min8(vint16 x) { x = vreduce_min4(x); return min(x, shuffle4<1,0,3,2>(x)); }
  __forceinline vint16 vreduce_min (vint16 x) { x = vreduce_min8(x); return min(x, shuffle4<2,3,0,1>(x)); }

  __forceinline vint16 vreduce_max2(vint16 x) {                      return max(x, shuffle<1,0,3,2>(x)); }
  __forceinline vint16 vreduce_max4(vint16 x) { x = vreduce_max2(x); return max(x, shuffle<2,3,0,1>(x)); }
  __forceinline vint16 vreduce_max8(vint16 x) { x = vreduce_max4(x); return max(x, shuffle4<1,0,3,2>(x)); }
  __forceinline vint16 vreduce_max (vint16 x) { x = vreduce_max8(x); return max(x, shuffle4<2,3,0,1>(x)); }

  __forceinline vint16 vreduce_and2(vint16 x) {                      return x & shuffle<1,0,3,2>(x); }
  __forceinline vint16 vreduce_and4(vint16 x) { x = vreduce_and2(x); return x & shuffle<2,3,0,1>(x); }
  __forceinline vint16 vreduce_and8(vint16 x) { x = vreduce_and4(x); return x & shuffle4<1,0,3,2>(x); }
  __forceinline vint16 vreduce_and (vint16 x) { x = vreduce_and8(x); return x & shuffle4<2,3,0,1>(x); }

  __forceinline vint16 vreduce_or2(vint16 x) {                     return x | shuffle<1,0,3,2>(x); }
  __forceinline vint16 vreduce_or4(vint16 x) { x = vreduce_or2(x); return x | shuffle<2,3,0,1>(x); }
  __forceinline vint16 vreduce_or8(vint16 x) { x = vreduce_or4(x); return x | shuffle4<1,0,3,2>(x); }
  __forceinline vint16 vreduce_or (vint16 x) { x = vreduce_or8(x); return x | shuffle4<2,3,0,1>(x); }

  __forceinline vint16 vreduce_add2(vint16 x) {                      return x + shuffle<1,0,3,2>(x); }
  __forceinline vint16 vreduce_add4(vint16 x) { x = vreduce_add2(x); return x + shuffle<2,3,0,1>(x); }
  __forceinline vint16 vreduce_add8(vint16 x) { x = vreduce_add4(x); return x + shuffle4<1,0,3,2>(x); }
  __forceinline vint16 vreduce_add (vint16 x) { x = vreduce_add8(x); return x + shuffle4<2,3,0,1>(x); }
  
  __forceinline int reduce_min(const vint16& v) { return toScalar(vreduce_min(v)); }
  __forceinline int reduce_max(const vint16& v) { return toScalar(vreduce_max(v)); }
  __forceinline int reduce_and(const vint16& v) { return toScalar(vreduce_and(v)); }
  __forceinline int reduce_or (const vint16& v) { return toScalar(vreduce_or (v)); }
  __forceinline int reduce_add(const vint16& v) { return toScalar(vreduce_add(v)); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vint16 conflict(const vint16& index)
  {
    return _mm512_conflict_epi32(index);
  }

  __forceinline vint16 conflict(const vboolf16& mask, vint16& dest, const vint16& index)
  {
    return _mm512_mask_conflict_epi32(dest,mask,index);
  }    

  __forceinline vint16 convert_uint32_t(const __m512& f) {
    return _mm512_cvtps_epu32(f);
  }

  __forceinline vint16 permute(vint16 v, vint16 index) {
    return _mm512_permutexvar_epi32(index,v);  
  }

  __forceinline vint16 reverse(const vint16 &a) {
    return permute(a,vint16(reverse_step));
  }

  __forceinline vint16 prefix_sum(const vint16& a) 
  {
    const vint16 z(zero);
    vint16 v = a;
    v = v + align_shift_right<16-1>(v,z);
    v = v + align_shift_right<16-2>(v,z);
    v = v + align_shift_right<16-4>(v,z);
    v = v + align_shift_right<16-8>(v,z);
    return v;  
  }

  __forceinline vint16 reverse_prefix_sum(const vint16& a) 
  {
    const vint16 z(zero);
    vint16 v = a;
    v = v + align_shift_right<1>(z,v);
    v = v + align_shift_right<2>(z,v);
    v = v + align_shift_right<4>(z,v);
    v = v + align_shift_right<8>(z,v);
    return v;  
  }

  /* this should use a vbool8 and a vint8_64...*/
  template<int scale = 1, int hint = _MM_HINT_T0>
    __forceinline void gather_prefetch64(const void* base_addr, const vbool16& mask, const vint16& offset)
  {
#if defined(__AVX512PF__)
    _mm512_mask_prefetch_i64gather_pd(offset, mask, base_addr, scale, hint);
#endif
  }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline embree_ostream operator <<(embree_ostream cout, const vint16& v)
  {
    cout << "<" << v[0];
    for (int i=1; i<16; i++) cout << ", " << v[i];
    cout << ">";
    return cout;
  }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
