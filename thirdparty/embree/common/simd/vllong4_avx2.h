// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

namespace embree
{ 
  /* 4-wide AVX2 64-bit long long type */
  template<>
  struct vllong<4>
  {
    typedef vboold4 Bool;

    enum  { size = 4 }; // number of SIMD elements
    union {             // data
      __m256i v; 
      long long i[4];
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
       
    __forceinline vllong() {}
    __forceinline vllong(const vllong4& t) { v = t.v; }
    __forceinline vllong4& operator =(const vllong4& f) { v = f.v; return *this; }

    __forceinline vllong(const __m256i& t) { v = t; }
    __forceinline operator __m256i() const { return v; }
    __forceinline operator __m256d() const { return _mm256_castsi256_pd(v); }


    __forceinline vllong(long long i) {
      v = _mm256_set1_epi64x(i);
    }
    
    __forceinline vllong(long long a, long long b, long long c, long long d) {
      v = _mm256_set_epi64x(d,c,b,a);      
    }
   
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vllong(ZeroTy) : v(_mm256_setzero_si256()) {}
    __forceinline vllong(OneTy)  : v(_mm256_set1_epi64x(1)) {}
    __forceinline vllong(StepTy) : v(_mm256_set_epi64x(3,2,1,0)) {}
    __forceinline vllong(ReverseStepTy) : v(_mm256_set_epi64x(0,1,2,3)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(void* __restrict__ ptr, const vllong4& a) {
      _mm256_stream_ps((float*)ptr,_mm256_castsi256_ps(a));
    }

    static __forceinline vllong4 loadu(const void* addr)
    {
      return _mm256_loadu_si256((__m256i*)addr);
    }

    static __forceinline vllong4 load(const vllong4* addr) {
      return _mm256_load_si256((__m256i*)addr);
    }

    static __forceinline vllong4 load(const long long* addr) {
      return _mm256_load_si256((__m256i*)addr);
    }

    static __forceinline void store(void* ptr, const vllong4& v) {
      _mm256_store_si256((__m256i*)ptr,v);
    }

    static __forceinline void storeu(void* ptr, const vllong4& v) {
      _mm256_storeu_si256((__m256i*)ptr,v);
    }

    static __forceinline void storeu(const vboold4& mask, long long* ptr, const vllong4& f) {
#if defined(__AVX512VL__)
      _mm256_mask_storeu_epi64(ptr,mask,f);
#else
      _mm256_maskstore_pd((double*)ptr,mask,_mm256_castsi256_pd(f));
#endif
    }

    static __forceinline void store(const vboold4& mask, void* ptr, const vllong4& f) {
#if defined(__AVX512VL__)
      _mm256_mask_store_epi64(ptr,mask,f);
#else
      _mm256_maskstore_pd((double*)ptr,mask,_mm256_castsi256_pd(f));
#endif
    }

    static __forceinline vllong4 broadcast64bit(size_t v) {
      return _mm256_set1_epi64x(v);
    }

    static __forceinline size_t extract64bit(const vllong4& v)
    {
      return _mm_cvtsi128_si64(_mm256_castsi256_si128(v));
    }


    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline       long long& operator [](size_t index)       { assert(index < 4); return i[index]; }
    __forceinline const long long& operator [](size_t index) const { assert(index < 4); return i[index]; }

  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Select
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vllong4 select(const vboold4& m, const vllong4& t, const vllong4& f) {
  #if defined(__AVX512VL__)
    return _mm256_mask_blend_epi64(m, f, t);
  #else
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(f), _mm256_castsi256_pd(t), m));
  #endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold4 asBool(const vllong4& a) { return _mm256_movepi64_mask(a); }
#else
  __forceinline vboold4 asBool(const vllong4& a) { return _mm256_castsi256_pd(a); }
#endif

  __forceinline vllong4 operator +(const vllong4& a) { return a; }
  __forceinline vllong4 operator -(const vllong4& a) { return _mm256_sub_epi64(_mm256_setzero_si256(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong4 operator +(const vllong4& a, const vllong4& b) { return _mm256_add_epi64(a, b); }
  __forceinline vllong4 operator +(const vllong4& a, long long      b) { return a + vllong4(b); }
  __forceinline vllong4 operator +(long long      a, const vllong4& b) { return vllong4(a) + b; }

  __forceinline vllong4 operator -(const vllong4& a, const vllong4& b) { return _mm256_sub_epi64(a, b); }
  __forceinline vllong4 operator -(const vllong4& a, long long      b) { return a - vllong4(b); }
  __forceinline vllong4 operator -(long long      a, const vllong4& b) { return vllong4(a) - b; }

  /* only low 32bit part */
  __forceinline vllong4 operator *(const vllong4& a, const vllong4& b) { return _mm256_mul_epi32(a, b); }
  __forceinline vllong4 operator *(const vllong4& a, long long      b) { return a * vllong4(b); }
  __forceinline vllong4 operator *(long long      a, const vllong4& b) { return vllong4(a) * b; }

  __forceinline vllong4 operator &(const vllong4& a, const vllong4& b) { return _mm256_and_si256(a, b); }
  __forceinline vllong4 operator &(const vllong4& a, long long      b) { return a & vllong4(b); }
  __forceinline vllong4 operator &(long long      a, const vllong4& b) { return vllong4(a) & b; }

  __forceinline vllong4 operator |(const vllong4& a, const vllong4& b) { return _mm256_or_si256(a, b); }
  __forceinline vllong4 operator |(const vllong4& a, long long      b) { return a | vllong4(b); }
  __forceinline vllong4 operator |(long long      a, const vllong4& b) { return vllong4(a) | b; }

  __forceinline vllong4 operator ^(const vllong4& a, const vllong4& b) { return _mm256_xor_si256(a, b); }
  __forceinline vllong4 operator ^(const vllong4& a, long long      b) { return a ^ vllong4(b); }
  __forceinline vllong4 operator ^(long long      a, const vllong4& b) { return vllong4(a) ^ b; }

  __forceinline vllong4 operator <<(const vllong4& a, long long n) { return _mm256_slli_epi64(a, (int)n); }
  //__forceinline vllong4 operator >>(const vllong4& a, long long n) { return _mm256_srai_epi64(a, n); }

  __forceinline vllong4 operator <<(const vllong4& a, const vllong4& n) { return _mm256_sllv_epi64(a, n); }
  //__forceinline vllong4 operator >>(const vllong4& a, const vllong4& n) { return _mm256_srav_epi64(a, n); }
  //__forceinline vllong4 sra(const vllong4& a, long long b) { return _mm256_srai_epi64(a, b); }

  __forceinline vllong4 srl(const vllong4& a, long long b) { return _mm256_srli_epi64(a, (int)b); }
  
  //__forceinline vllong4 min(const vllong4& a, const vllong4& b) { return _mm256_min_epi64(a, b); }
  //__forceinline vllong4 min(const vllong4& a, long long      b) { return min(a,vllong4(b)); }
  //__forceinline vllong4 min(long long      a, const vllong4& b) { return min(vllong4(a),b); }

  //__forceinline vllong4 max(const vllong4& a, const vllong4& b) { return _mm256_max_epi64(a, b); }
  //__forceinline vllong4 max(const vllong4& a, long long      b) { return max(a,vllong4(b)); }
  //__forceinline vllong4 max(long long      a, const vllong4& b) { return max(vllong4(a),b); }

#if defined(__AVX512VL__)
  __forceinline vllong4 mask_and(const vboold4& m, const vllong4& c, const vllong4& a, const vllong4& b) { return _mm256_mask_and_epi64(c,m,a,b); }
  __forceinline vllong4 mask_or (const vboold4& m, const vllong4& c, const vllong4& a, const vllong4& b) { return _mm256_mask_or_epi64(c,m,a,b); }
#else
  __forceinline vllong4 mask_and(const vboold4& m, const vllong4& c, const vllong4& a, const vllong4& b) { return select(m, a & b, c); }
  __forceinline vllong4 mask_or (const vboold4& m, const vllong4& c, const vllong4& a, const vllong4& b) { return select(m, a | b, c); }
#endif
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vllong4& operator +=(vllong4& a, const vllong4& b) { return a = a + b; }
  __forceinline vllong4& operator +=(vllong4& a, long long      b) { return a = a + b; }
  
  __forceinline vllong4& operator -=(vllong4& a, const vllong4& b) { return a = a - b; }
  __forceinline vllong4& operator -=(vllong4& a, long long      b) { return a = a - b; }

  __forceinline vllong4& operator *=(vllong4& a, const vllong4& b) { return a = a * b; }
  __forceinline vllong4& operator *=(vllong4& a, long long      b) { return a = a * b; }
  
  __forceinline vllong4& operator &=(vllong4& a, const vllong4& b) { return a = a & b; }
  __forceinline vllong4& operator &=(vllong4& a, long long      b) { return a = a & b; }
  
  __forceinline vllong4& operator |=(vllong4& a, const vllong4& b) { return a = a | b; }
  __forceinline vllong4& operator |=(vllong4& a, long long      b) { return a = a | b; }
  
  __forceinline vllong4& operator <<=(vllong4& a, long long      b) { return a = a << b; }
  //__forceinline vllong4& operator >>=(vllong4& a, long long      b) { return a = a >> b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold4 operator ==(const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboold4 operator !=(const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboold4 operator < (const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboold4 operator >=(const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboold4 operator > (const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboold4 operator <=(const vllong4& a, const vllong4& b) { return _mm256_cmp_epi64_mask(a,b,_MM_CMPINT_LE); }
#else
  __forceinline vboold4 operator ==(const vllong4& a, const vllong4& b) { return _mm256_cmpeq_epi64(a,b); }
  __forceinline vboold4 operator !=(const vllong4& a, const vllong4& b) { return !(a == b); }
  __forceinline vboold4 operator > (const vllong4& a, const vllong4& b) { return _mm256_cmpgt_epi64(a,b); }
  __forceinline vboold4 operator < (const vllong4& a, const vllong4& b) { return _mm256_cmpgt_epi64(b,a); }
  __forceinline vboold4 operator >=(const vllong4& a, const vllong4& b) { return !(a < b); }
  __forceinline vboold4 operator <=(const vllong4& a, const vllong4& b) { return !(a > b); }
#endif

  __forceinline vboold4 operator ==(const vllong4& a, long long      b) { return a == vllong4(b); }
  __forceinline vboold4 operator ==(long long      a, const vllong4& b) { return vllong4(a) == b; }

  __forceinline vboold4 operator !=(const vllong4& a, long long      b) { return a != vllong4(b); }
  __forceinline vboold4 operator !=(long long      a, const vllong4& b) { return vllong4(a) != b; }

  __forceinline vboold4 operator > (const vllong4& a, long long      b) { return a >  vllong4(b); }
  __forceinline vboold4 operator > (long long      a, const vllong4& b) { return vllong4(a) >  b; }

  __forceinline vboold4 operator < (const vllong4& a, long long      b) { return a <  vllong4(b); }
  __forceinline vboold4 operator < (long long      a, const vllong4& b) { return vllong4(a) <  b; }

  __forceinline vboold4 operator >=(const vllong4& a, long long      b) { return a >= vllong4(b); }
  __forceinline vboold4 operator >=(long long      a, const vllong4& b) { return vllong4(a) >= b; }

  __forceinline vboold4 operator <=(const vllong4& a, long long      b) { return a <= vllong4(b); }
  __forceinline vboold4 operator <=(long long      a, const vllong4& b) { return vllong4(a) <= b; }

  __forceinline vboold4 eq(const vllong4& a, const vllong4& b) { return a == b; }
  __forceinline vboold4 ne(const vllong4& a, const vllong4& b) { return a != b; }
  __forceinline vboold4 lt(const vllong4& a, const vllong4& b) { return a <  b; }
  __forceinline vboold4 ge(const vllong4& a, const vllong4& b) { return a >= b; }
  __forceinline vboold4 gt(const vllong4& a, const vllong4& b) { return a >  b; }
  __forceinline vboold4 le(const vllong4& a, const vllong4& b) { return a <= b; }

#if defined(__AVX512VL__)
  __forceinline vboold4 eq(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_EQ); }
  __forceinline vboold4 ne(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_NE); }
  __forceinline vboold4 lt(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_LT); }
  __forceinline vboold4 ge(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_GE); }
  __forceinline vboold4 gt(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_GT); }
  __forceinline vboold4 le(const vboold4& mask, const vllong4& a, const vllong4& b) { return _mm256_mask_cmp_epi64_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboold4 eq(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a == b); }
  __forceinline vboold4 ne(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a != b); }
  __forceinline vboold4 lt(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a <  b); }
  __forceinline vboold4 ge(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a >= b); }
  __forceinline vboold4 gt(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a >  b); }
  __forceinline vboold4 le(const vboold4& mask, const vllong4& a, const vllong4& b) { return mask & (a <= b); }
#endif

  __forceinline void xchg(const vboold4& m, vllong4& a, vllong4& b) {
    const vllong4 c = a; a = select(m,b,a); b = select(m,c,b);
  }

  __forceinline vboold4 test(const vllong4& a, const vllong4& b) {
#if defined(__AVX512VL__)
    return _mm256_test_epi64_mask(a,b);
#else
    return _mm256_testz_si256(a,b);
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i0, int i1>
  __forceinline vllong4 shuffle(const vllong4& v) {
    return _mm256_castpd_si256(_mm256_permute_pd(_mm256_castsi256_pd(v), (i1 << 3) | (i0 << 2) | (i1 << 1) | i0));
  }

  template<int i>
  __forceinline vllong4 shuffle(const vllong4& v) {
    return shuffle<i, i>(v);
  }

  template<int i0, int i1>
  __forceinline vllong4 shuffle2(const vllong4& v) {
    return _mm256_castpd_si256(_mm256_permute2f128_pd(_mm256_castsi256_pd(v), _mm256_castsi256_pd(v), (i1 << 4) | i0));
  }

  __forceinline long long toScalar(const vllong4& v) {
    return _mm_cvtsi128_si64(_mm256_castsi256_si128(v));
  }

#if defined(__AVX512VL__)
  __forceinline vllong4 permute(const vllong4& a, const __m256i& index) {
    // workaround for GCC 7.x
#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
    return _mm256_permutex2var_epi64(a,index,a);
#else
    return _mm256_permutexvar_epi64(index,a);
#endif
  }

  __forceinline vllong4 permutex2var(const vllong4& index, const vllong4& a, const vllong4& b) {
    return _mm256_permutex2var_epi64(a,index,b);
  }

#endif
  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////
  

  __forceinline vllong4 vreduce_and2(const vllong4& x) { return x & shuffle<1,0>(x); }
  __forceinline vllong4 vreduce_and (const vllong4& y) { const vllong4 x = vreduce_and2(y); return x & shuffle2<1,0>(x); }

  __forceinline vllong4 vreduce_or2(const vllong4& x) { return x | shuffle<1,0>(x); }
  __forceinline vllong4 vreduce_or (const vllong4& y) { const vllong4 x = vreduce_or2(y); return x | shuffle2<1,0>(x); }

  __forceinline vllong4 vreduce_add2(const vllong4& x) { return x + shuffle<1,0>(x); }
  __forceinline vllong4 vreduce_add (const vllong4& y) { const vllong4 x = vreduce_add2(y); return x + shuffle2<1,0>(x); }

  __forceinline long long reduce_add(const vllong4& a) { return toScalar(vreduce_add(a)); }
  __forceinline long long reduce_or (const vllong4& a) { return toScalar(vreduce_or(a)); }
  __forceinline long long reduce_and(const vllong4& a) { return toScalar(vreduce_and(a)); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline std::ostream& operator <<(std::ostream& cout, const vllong4& v)
  {
    cout << "<" << v[0];
    for (size_t i=1; i<4; i++) cout << ", " << v[i];
    cout << ">";
    return cout;
  }
}
