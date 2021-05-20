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
  /* 16-wide AVX-512 unsigned integer type */
  template<>
  struct vuint<16>
  {
    ALIGNED_STRUCT_(64);   

    typedef vboolf16 Bool;
    typedef vuint16  UInt;
    typedef vfloat16 Float;

    enum  { size = 16 }; // number of SIMD elements
    union {              // data
      __m512i v; 
      unsigned int i[16]; 
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
       
    __forceinline vuint() {}
    __forceinline vuint(const vuint16& t) { v = t.v; }
    __forceinline vuint16& operator =(const vuint16& f) { v = f.v; return *this; }

    __forceinline vuint(const __m512i& t) { v = t; }
    __forceinline operator __m512i() const { return v; }
    __forceinline operator __m256i() const { return _mm512_castsi512_si256(v); }

    __forceinline vuint(unsigned int i) {
      v = _mm512_set1_epi32(i);
    }

    __forceinline vuint(const vuint4& i) {
      v = _mm512_broadcast_i32x4(i);
    }

    __forceinline vuint(const vuint8& i) {
      v = _mm512_castps_si512(_mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castsi256_pd(i))));
    }
    
    __forceinline vuint(unsigned int a, unsigned int b, unsigned int c, unsigned int d) {
      v = _mm512_set4_epi32(d,c,b,a);      
    }

    __forceinline vuint(unsigned int a0 , unsigned int a1 , unsigned int a2 , unsigned int a3,
                        unsigned int a4 , unsigned int a5 , unsigned int a6 , unsigned int a7,
                        unsigned int a8 , unsigned int a9 , unsigned int a10, unsigned int a11,
                        unsigned int a12, unsigned int a13, unsigned int a14, unsigned int a15)
    {
      v = _mm512_set_epi32(a15,a14,a13,a12,a11,a10,a9,a8,a7,a6,a5,a4,a3,a2,a1,a0);
    }
   
    __forceinline explicit vuint(const __m512& f) {
      v = _mm512_cvtps_epu32(f);
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vuint(ZeroTy) : v(_mm512_setzero_epi32()) {}
    __forceinline vuint(OneTy)  : v(_mm512_set1_epi32(1)) {}
    __forceinline vuint(StepTy) : v(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)) {}
    __forceinline vuint(ReverseStepTy) : v(_mm512_setr_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(void* __restrict__ ptr, const vuint16& a) {
      _mm512_stream_si512((__m512i*)ptr,a);
    }

    static __forceinline vuint16 loadu(const void* addr)
    {
      return _mm512_loadu_si512(addr);
    }

    static __forceinline vuint16 loadu(const unsigned char* ptr) { return _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)ptr)); }
    static __forceinline vuint16 loadu(const unsigned short* ptr) { return _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)ptr)); }

    static __forceinline vuint16 load(const vuint16* addr) {
      return _mm512_load_si512(addr);
    }

    static __forceinline vuint16 load(const unsigned int* addr) {
      return _mm512_load_si512(addr);
    }

    static __forceinline vuint16 load(unsigned short* ptr) { return _mm512_cvtepu16_epi32(*(__m256i*)ptr); }


    static __forceinline void store(void* ptr, const vuint16& v) {
      _mm512_store_si512(ptr,v);
    }

    static __forceinline void storeu(void* ptr, const vuint16& v) {
      _mm512_storeu_si512(ptr,v);
    }

    static __forceinline void storeu(const vboolf16& mask, void* ptr, const vuint16& f) {
      _mm512_mask_storeu_epi32(ptr,mask,f);
    }

    static __forceinline void store(const vboolf16& mask, void* addr, const vuint16& v2) {
      _mm512_mask_store_epi32(addr,mask,v2);
    }

    static __forceinline vuint16 compact(const vboolf16& mask, vuint16& v) {
      return _mm512_mask_compress_epi32(v,mask,v);
    }

    static __forceinline vuint16 compact(const vboolf16& mask, const vuint16& a, vuint16& b) {
      return _mm512_mask_compress_epi32(a,mask,b);
    }

    static __forceinline vuint16 expand(const vboolf16& mask, const vuint16& a, vuint16& b) {
      return _mm512_mask_expand_epi32(b,mask,a);
    }

    template<int scale = 4>
    static __forceinline vuint16 gather(const unsigned int* ptr, const vint16& index) {
      return _mm512_i32gather_epi32(index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline vuint16 gather(const vboolf16& mask, const unsigned int* ptr, const vint16& index) {
      return _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(),mask,index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline vuint16 gather(const vboolf16& mask, vuint16& dest, const unsigned int* ptr, const vint16& index) {
      return _mm512_mask_i32gather_epi32(dest,mask,index,ptr,scale);
    }

    template<int scale = 4>
    static __forceinline void scatter(unsigned int* ptr, const vint16& index, const vuint16& v) {
      _mm512_i32scatter_epi32((int*)ptr,index,v,scale);
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf16& mask, unsigned int* ptr, const vint16& index, const vuint16& v) {
      _mm512_mask_i32scatter_epi32((int*)ptr,mask,index,v,scale);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline       unsigned int& operator [](size_t index)       { assert(index < 16); return i[index]; }
    __forceinline const unsigned int& operator [](size_t index) const { assert(index < 16); return i[index]; }

    __forceinline unsigned int uint    (size_t index) const { assert(index < 16); return ((unsigned int*)i)[index]; }
    __forceinline size_t&      uint64_t(size_t index) const { assert(index < 8);  return ((size_t*)i)[index]; }
  };
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf16 asBool(const vuint16& a) { return _mm512_movepi32_mask(a); }

  __forceinline vuint16 operator +(const vuint16& a) { return a; }
  __forceinline vuint16 operator -(const vuint16& a) { return _mm512_sub_epi32(_mm512_setzero_epi32(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint16 operator +(const vuint16& a, const vuint16& b) { return _mm512_add_epi32(a, b); }
  __forceinline vuint16 operator +(const vuint16& a, unsigned int   b) { return a + vuint16(b); }
  __forceinline vuint16 operator +(unsigned int   a, const vuint16& b) { return vuint16(a) + b; }

  __forceinline vuint16 operator -(const vuint16& a, const vuint16& b) { return _mm512_sub_epi32(a, b); }
  __forceinline vuint16 operator -(const vuint16& a, unsigned int   b) { return a - vuint16(b); }
  __forceinline vuint16 operator -(unsigned int   a, const vuint16& b) { return vuint16(a) - b; }

  __forceinline vuint16 operator *(const vuint16& a, const vuint16& b) { return _mm512_mul_epu32(a, b); }
  __forceinline vuint16 operator *(const vuint16& a, unsigned int   b) { return a * vuint16(b); }
  __forceinline vuint16 operator *(unsigned int   a, const vuint16& b) { return vuint16(a) * b; }

  __forceinline vuint16 operator &(const vuint16& a, const vuint16& b) { return _mm512_and_epi32(a, b); }
  __forceinline vuint16 operator &(const vuint16& a, unsigned int   b) { return a & vuint16(b); }
  __forceinline vuint16 operator &(unsigned int   a, const vuint16& b) { return vuint16(a) & b; }

  __forceinline vuint16 operator |(const vuint16& a, const vuint16& b) { return _mm512_or_epi32(a, b); }
  __forceinline vuint16 operator |(const vuint16& a, unsigned int   b) { return a | vuint16(b); }
  __forceinline vuint16 operator |(unsigned int   a, const vuint16& b) { return vuint16(a) | b; }

  __forceinline vuint16 operator ^(const vuint16& a, const vuint16& b) { return _mm512_xor_epi32(a, b); }
  __forceinline vuint16 operator ^(const vuint16& a, unsigned int   b) { return a ^ vuint16(b); }
  __forceinline vuint16 operator ^(unsigned int   a, const vuint16& b) { return vuint16(a) ^ b; }

  __forceinline vuint16 operator <<(const vuint16& a, unsigned int n) { return _mm512_slli_epi32(a, n); }
  __forceinline vuint16 operator >>(const vuint16& a, unsigned int n) { return _mm512_srli_epi32(a, n); }

  __forceinline vuint16 operator <<(const vuint16& a, const vuint16& n) { return _mm512_sllv_epi32(a, n); }
  __forceinline vuint16 operator >>(const vuint16& a, const vuint16& n) { return _mm512_srlv_epi32(a, n); }

  __forceinline vuint16 sll (const vuint16& a, unsigned int b) { return _mm512_slli_epi32(a, b); }
  __forceinline vuint16 sra (const vuint16& a, unsigned int b) { return _mm512_srai_epi32(a, b); }
  __forceinline vuint16 srl (const vuint16& a, unsigned int b) { return _mm512_srli_epi32(a, b); }
  
  __forceinline vuint16 min(const vuint16& a, const vuint16& b) { return _mm512_min_epu32(a, b); }
  __forceinline vuint16 min(const vuint16& a, unsigned int   b) { return min(a,vuint16(b)); }
  __forceinline vuint16 min(unsigned int   a, const vuint16& b) { return min(vuint16(a),b); }

  __forceinline vuint16 max(const vuint16& a, const vuint16& b) { return _mm512_max_epu32(a, b); }
  __forceinline vuint16 max(const vuint16& a, unsigned int   b) { return max(a,vuint16(b)); }
  __forceinline vuint16 max(unsigned int   a, const vuint16& b) { return max(vuint16(a),b); }
  
  __forceinline vuint16 mask_add(const vboolf16& mask, vuint16& c, const vuint16& a, const vuint16& b) { return _mm512_mask_add_epi32(c,mask,a,b); }
  __forceinline vuint16 mask_sub(const vboolf16& mask, vuint16& c, const vuint16& a, const vuint16& b) { return _mm512_mask_sub_epi32(c,mask,a,b); }

  __forceinline vuint16 mask_and(const vboolf16& m, vuint16& c, const vuint16& a, const vuint16& b) { return _mm512_mask_and_epi32(c,m,a,b); }
  __forceinline vuint16 mask_or (const vboolf16& m, vuint16& c, const vuint16& a, const vuint16& b) { return _mm512_mask_or_epi32(c,m,a,b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint16& operator +=(vuint16& a, const vuint16& b) { return a = a + b; }
  __forceinline vuint16& operator +=(vuint16& a, unsigned int   b) { return a = a + b; }
  
  __forceinline vuint16& operator -=(vuint16& a, const vuint16& b) { return a = a - b; }
  __forceinline vuint16& operator -=(vuint16& a, unsigned int   b) { return a = a - b; }

  __forceinline vuint16& operator *=(vuint16& a, const vuint16& b) { return a = a * b; }
  __forceinline vuint16& operator *=(vuint16& a, unsigned int   b) { return a = a * b; }
  
  __forceinline vuint16& operator &=(vuint16& a, const vuint16& b) { return a = a & b; }
  __forceinline vuint16& operator &=(vuint16& a, unsigned int   b) { return a = a & b; }
  
  __forceinline vuint16& operator |=(vuint16& a, const vuint16& b) { return a = a | b; }
  __forceinline vuint16& operator |=(vuint16& a, unsigned int   b) { return a = a | b; }
  
  __forceinline vuint16& operator <<=(vuint16& a, unsigned int b) { return a = a << b; }
  __forceinline vuint16& operator >>=(vuint16& a, unsigned int b) { return a = a >> b; }


  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf16 operator ==(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 operator ==(const vuint16& a, unsigned int   b) { return a == vuint16(b); }
  __forceinline vboolf16 operator ==(unsigned int   a, const vuint16& b) { return vuint16(a) == b; }
  
  __forceinline vboolf16 operator !=(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 operator !=(const vuint16& a, unsigned int   b) { return a != vuint16(b); }
  __forceinline vboolf16 operator !=(unsigned int   a, const vuint16& b) { return vuint16(a) != b; }
  
  __forceinline vboolf16 operator < (const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 operator < (const vuint16& a, unsigned int   b) { return a <  vuint16(b); }
  __forceinline vboolf16 operator < (unsigned int   a, const vuint16& b) { return vuint16(a) <  b; }
  
  __forceinline vboolf16 operator >=(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 operator >=(const vuint16& a, unsigned int   b) { return a >= vuint16(b); }
  __forceinline vboolf16 operator >=(unsigned int   a, const vuint16& b) { return vuint16(a) >= b; }

  __forceinline vboolf16 operator > (const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 operator > (const vuint16& a, unsigned int   b) { return a >  vuint16(b); }
  __forceinline vboolf16 operator > (unsigned int   a, const vuint16& b) { return vuint16(a) >  b; }

  __forceinline vboolf16 operator <=(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_LE); }
  __forceinline vboolf16 operator <=(const vuint16& a, unsigned int   b) { return a <= vuint16(b); }
  __forceinline vboolf16 operator <=(unsigned int   a, const vuint16& b) { return vuint16(a) <= b; }

  __forceinline vboolf16 eq(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 ne(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 lt(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 ge(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 gt(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 le(const vuint16& a, const vuint16& b) { return _mm512_cmp_epu32_mask(a,b,_MM_CMPINT_LE); }

  __forceinline vboolf16 eq(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_EQ); }
  __forceinline vboolf16 ne(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_NE); }
  __forceinline vboolf16 lt(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_LT); }
  __forceinline vboolf16 ge(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_GE); }
  __forceinline vboolf16 gt(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_GT); }
  __forceinline vboolf16 le(const vboolf16 mask, const vuint16& a, const vuint16& b) { return _mm512_mask_cmp_epu32_mask(mask,a,b,_MM_CMPINT_LE); }
    
 
  __forceinline vuint16 select(const vboolf16& m, const vuint16& t, const vuint16& f) {
    return _mm512_mask_or_epi32(f,m,t,t); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i>
  __forceinline vuint16 shuffle(const vuint16& v) {
    return _mm512_castps_si512(_mm512_permute_ps(_mm512_castsi512_ps(v), _MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vuint16 shuffle(const vuint16& v) {
    return _mm512_castps_si512(_mm512_permute_ps(_mm512_castsi512_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i>
  __forceinline vuint16 shuffle4(const vuint16& v) {
    return _mm512_castps_si512(_mm512_shuffle_f32x4(_mm512_castsi512_ps(v), _mm512_castsi512_ps(v) ,_MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vuint16 shuffle4(const vuint16& v) {
    return _mm512_castps_si512(_mm512_shuffle_f32x4(_mm512_castsi512_ps(v), _mm512_castsi512_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i>
  __forceinline vuint16 align_shift_right(const vuint16& a, const vuint16& b) {
    return _mm512_alignr_epi32(a, b, i);
  };

  __forceinline unsigned int toScalar(const vuint16& v) {
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(v));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint16 vreduce_min2(vuint16 x) {                      return min(x, shuffle<1,0,3,2>(x)); }
  __forceinline vuint16 vreduce_min4(vuint16 x) { x = vreduce_min2(x); return min(x, shuffle<2,3,0,1>(x)); }
  __forceinline vuint16 vreduce_min8(vuint16 x) { x = vreduce_min4(x); return min(x, shuffle4<1,0,3,2>(x)); }
  __forceinline vuint16 vreduce_min (vuint16 x) { x = vreduce_min8(x); return min(x, shuffle4<2,3,0,1>(x)); }

  __forceinline vuint16 vreduce_max2(vuint16 x) {                      return max(x, shuffle<1,0,3,2>(x)); }
  __forceinline vuint16 vreduce_max4(vuint16 x) { x = vreduce_max2(x); return max(x, shuffle<2,3,0,1>(x)); }
  __forceinline vuint16 vreduce_max8(vuint16 x) { x = vreduce_max4(x); return max(x, shuffle4<1,0,3,2>(x)); }
  __forceinline vuint16 vreduce_max (vuint16 x) { x = vreduce_max8(x); return max(x, shuffle4<2,3,0,1>(x)); }

  __forceinline vuint16 vreduce_and2(vuint16 x) {                      return x & shuffle<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_and4(vuint16 x) { x = vreduce_and2(x); return x & shuffle<2,3,0,1>(x); }
  __forceinline vuint16 vreduce_and8(vuint16 x) { x = vreduce_and4(x); return x & shuffle4<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_and (vuint16 x) { x = vreduce_and8(x); return x & shuffle4<2,3,0,1>(x); }

  __forceinline vuint16 vreduce_or2(vuint16 x) {                     return x | shuffle<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_or4(vuint16 x) { x = vreduce_or2(x); return x | shuffle<2,3,0,1>(x); }
  __forceinline vuint16 vreduce_or8(vuint16 x) { x = vreduce_or4(x); return x | shuffle4<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_or (vuint16 x) { x = vreduce_or8(x); return x | shuffle4<2,3,0,1>(x); }

  __forceinline vuint16 vreduce_add2(vuint16 x) {                      return x + shuffle<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_add4(vuint16 x) { x = vreduce_add2(x); return x + shuffle<2,3,0,1>(x); }
  __forceinline vuint16 vreduce_add8(vuint16 x) { x = vreduce_add4(x); return x + shuffle4<1,0,3,2>(x); }
  __forceinline vuint16 vreduce_add (vuint16 x) { x = vreduce_add8(x); return x + shuffle4<2,3,0,1>(x); }

  __forceinline unsigned int reduce_min(const vuint16& v) { return toScalar(vreduce_min(v)); }
  __forceinline unsigned int reduce_max(const vuint16& v) { return toScalar(vreduce_max(v)); }
  __forceinline unsigned int reduce_and(const vuint16& v) { return toScalar(vreduce_and(v)); }
  __forceinline unsigned int reduce_or (const vuint16& v) { return toScalar(vreduce_or (v)); }
  __forceinline unsigned int reduce_add(const vuint16& v) { return toScalar(vreduce_add(v)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vuint16 permute(vuint16 v, vuint16 index) {
    return _mm512_permutexvar_epi32(index,v);  
  }

  __forceinline vuint16 reverse(const vuint16& a) {
    return permute(a,vuint16(reverse_step));
  }

  __forceinline vuint16 prefix_sum(const vuint16& a) 
  {
    const vuint16 z(zero);
    vuint16 v = a;
    v = v + align_shift_right<16-1>(v,z);
    v = v + align_shift_right<16-2>(v,z);
    v = v + align_shift_right<16-4>(v,z);
    v = v + align_shift_right<16-8>(v,z);
    return v;  
  }

  __forceinline vuint16 reverse_prefix_sum(const vuint16& a) 
  {
    const vuint16 z(zero);
    vuint16 v = a;
    v = v + align_shift_right<1>(z,v);
    v = v + align_shift_right<2>(z,v);
    v = v + align_shift_right<4>(z,v);
    v = v + align_shift_right<8>(z,v);
    return v;  
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline embree_ostream operator <<(embree_ostream cout, const vuint16& v)
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
