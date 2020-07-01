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
  /* 4-wide AVX 64-bit double type */
  template<>
  struct vdouble<4>
  {
    typedef vboold4 Bool;

    enum  { size = 4 }; // number of SIMD elements
    union {             // data
      __m256d v; 
      double i[4]; 
    };
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
       
    __forceinline vdouble() {}
    __forceinline vdouble(const vdouble4& t) { v = t.v; }
    __forceinline vdouble4& operator =(const vdouble4& f) { v = f.v; return *this; }

    __forceinline vdouble(const __m256d& t) { v = t; }
    __forceinline operator __m256d() const { return v; }

    __forceinline vdouble(double i) {
      v = _mm256_set1_pd(i);
    }
    
    __forceinline vdouble(double a, double b, double c, double d) {
      v = _mm256_set_pd(d,c,b,a);      
    }
   
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vdouble(ZeroTy) : v(_mm256_setzero_pd()) {}
    __forceinline vdouble(OneTy)  : v(_mm256_set1_pd(1)) {}
    __forceinline vdouble(StepTy) : v(_mm256_set_pd(3.0,2.0,1.0,0.0)) {}
    __forceinline vdouble(ReverseStepTy) : v(_mm256_setr_pd(3.0,2.0,1.0,0.0)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(double *__restrict__ ptr, const vdouble4& a) {
      _mm256_stream_pd(ptr, a);
    }

    static __forceinline vdouble4 loadu(const double* addr) {
      return _mm256_loadu_pd(addr);
    }

    static __forceinline vdouble4 load(const vdouble4* addr) {
      return _mm256_load_pd((double*)addr);
    }

    static __forceinline vdouble4 load(const double* addr) {
      return _mm256_load_pd(addr);
    }

    static __forceinline void store(double* ptr, const vdouble4& v) {
      _mm256_store_pd(ptr, v);
    }

    static __forceinline void storeu(double* ptr, const vdouble4& v) {
      _mm256_storeu_pd(ptr, v);
    }

    static __forceinline vdouble4 broadcast(const void* a) { return _mm256_set1_pd(*(double*)a); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline       double& operator [](size_t index)       { assert(index < 4); return i[index]; }
    __forceinline const double& operator [](size_t index) const { assert(index < 4); return i[index]; }
  };
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX2__)
  __forceinline vdouble4 asDouble(const vllong4&  a) { return _mm256_castsi256_pd(a); }
  __forceinline vllong4  asLLong (const vdouble4& a) { return _mm256_castpd_si256(a); }
#endif

  __forceinline vdouble4 operator +(const vdouble4& a) { return a; }
  __forceinline vdouble4 operator -(const vdouble4& a) { return _mm256_sub_pd(_mm256_setzero_pd(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble4 operator +(const vdouble4& a, const vdouble4& b) { return _mm256_add_pd(a, b); }
  __forceinline vdouble4 operator +(const vdouble4& a, double          b) { return a + vdouble4(b); }
  __forceinline vdouble4 operator +(double          a, const vdouble4& b) { return vdouble4(a) + b; }

  __forceinline vdouble4 operator -(const vdouble4& a, const vdouble4& b) { return _mm256_sub_pd(a, b); }
  __forceinline vdouble4 operator -(const vdouble4& a, double          b) { return a - vdouble4(b); }
  __forceinline vdouble4 operator -(double          a, const vdouble4& b) { return vdouble4(a) - b; }

  __forceinline vdouble4 operator *(const vdouble4& a, const vdouble4& b) { return _mm256_mul_pd(a, b); }
  __forceinline vdouble4 operator *(const vdouble4& a, double          b) { return a * vdouble4(b); }
  __forceinline vdouble4 operator *(double          a, const vdouble4& b) { return vdouble4(a) * b; }

  __forceinline vdouble4 operator &(const vdouble4& a, const vdouble4& b) { return _mm256_and_pd(a, b); }
  __forceinline vdouble4 operator &(const vdouble4& a, double          b) { return a & vdouble4(b); }
  __forceinline vdouble4 operator &(double          a, const vdouble4& b) { return vdouble4(a) & b; }

  __forceinline vdouble4 operator |(const vdouble4& a, const vdouble4& b) { return _mm256_or_pd(a, b); }
  __forceinline vdouble4 operator |(const vdouble4& a, double          b) { return a | vdouble4(b); }
  __forceinline vdouble4 operator |(double          a, const vdouble4& b) { return vdouble4(a) | b; }

  __forceinline vdouble4 operator ^(const vdouble4& a, const vdouble4& b) { return _mm256_xor_pd(a, b); }
  __forceinline vdouble4 operator ^(const vdouble4& a, double          b) { return a ^ vdouble4(b); }
  __forceinline vdouble4 operator ^(double          a, const vdouble4& b) { return vdouble4(a) ^ b; }
  
  __forceinline vdouble4 min(const vdouble4& a, const vdouble4& b) { return _mm256_min_pd(a, b); }
  __forceinline vdouble4 min(const vdouble4& a, double          b) { return min(a,vdouble4(b)); }
  __forceinline vdouble4 min(double          a, const vdouble4& b) { return min(vdouble4(a),b); }

  __forceinline vdouble4 max(const vdouble4& a, const vdouble4& b) { return _mm256_max_pd(a, b); }
  __forceinline vdouble4 max(const vdouble4& a, double          b) { return max(a,vdouble4(b)); }
  __forceinline vdouble4 max(double          a, const vdouble4& b) { return max(vdouble4(a),b); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__FMA__)
  __forceinline vdouble4 madd (const vdouble4& a, const vdouble4& b, const vdouble4& c) { return _mm256_fmadd_pd(a,b,c); }
  __forceinline vdouble4 msub (const vdouble4& a, const vdouble4& b, const vdouble4& c) { return _mm256_fmsub_pd(a,b,c); }
  __forceinline vdouble4 nmadd(const vdouble4& a, const vdouble4& b, const vdouble4& c) { return _mm256_fnmadd_pd(a,b,c); }
  __forceinline vdouble4 nmsub(const vdouble4& a, const vdouble4& b, const vdouble4& c) { return _mm256_fnmsub_pd(a,b,c); }
#else
  __forceinline vdouble4 madd (const vdouble4& a, const vdouble4& b, const vdouble4& c) { return a*b+c; }
  __forceinline vdouble4 msub (const vdouble4& a, const vdouble4& b, const vdouble4& c) { return a*b-c; }
  __forceinline vdouble4 nmadd(const vdouble4& a, const vdouble4& b, const vdouble4& c) { return -a*b+c;}
  __forceinline vdouble4 nmsub(const vdouble4& a, const vdouble4& b, const vdouble4& c) { return -a*b-c; }
#endif

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble4& operator +=(vdouble4& a, const vdouble4& b) { return a = a + b; }
  __forceinline vdouble4& operator +=(vdouble4& a, double          b) { return a = a + b; }
  
  __forceinline vdouble4& operator -=(vdouble4& a, const vdouble4& b) { return a = a - b; }
  __forceinline vdouble4& operator -=(vdouble4& a, double          b) { return a = a - b; }

  __forceinline vdouble4& operator *=(vdouble4& a, const vdouble4& b) { return a = a * b; }
  __forceinline vdouble4& operator *=(vdouble4& a, double          b) { return a = a * b; }
  
  __forceinline vdouble4& operator &=(vdouble4& a, const vdouble4& b) { return a = a & b; }
  __forceinline vdouble4& operator &=(vdouble4& a, double          b) { return a = a & b; }
  
  __forceinline vdouble4& operator |=(vdouble4& a, const vdouble4& b) { return a = a | b; }
  __forceinline vdouble4& operator |=(vdouble4& a, double          b) { return a = a | b; }
  

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

#if defined(__AVX512VL__)
  __forceinline vboold4 operator ==(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_EQ); }
  __forceinline vboold4 operator !=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_NE); }
  __forceinline vboold4 operator < (const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_LT); }
  __forceinline vboold4 operator >=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_GE); }
  __forceinline vboold4 operator > (const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_GT); }
  __forceinline vboold4 operator <=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd_mask(a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboold4 operator ==(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ);  }
  __forceinline vboold4 operator !=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_NEQ_UQ); }
  __forceinline vboold4 operator < (const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_LT_OS);  }
  __forceinline vboold4 operator >=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_NLT_US); }
  __forceinline vboold4 operator > (const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_NLE_US); }
  __forceinline vboold4 operator <=(const vdouble4& a, const vdouble4& b) { return _mm256_cmp_pd(a, b, _CMP_LE_OS);  }
#endif

  __forceinline vboold4 operator ==(const vdouble4& a, double          b) { return a == vdouble4(b); }
  __forceinline vboold4 operator ==(double          a, const vdouble4& b) { return vdouble4(a) == b; }

  __forceinline vboold4 operator !=(const vdouble4& a, double          b) { return a != vdouble4(b); }
  __forceinline vboold4 operator !=(double          a, const vdouble4& b) { return vdouble4(a) != b; }

  __forceinline vboold4 operator < (const vdouble4& a, double          b) { return a <  vdouble4(b); }
  __forceinline vboold4 operator < (double          a, const vdouble4& b) { return vdouble4(a) <  b; }

  __forceinline vboold4 operator >=(const vdouble4& a, double          b) { return a >= vdouble4(b); }
  __forceinline vboold4 operator >=(double          a, const vdouble4& b) { return vdouble4(a) >= b; }

  __forceinline vboold4 operator > (const vdouble4& a, double          b) { return a >  vdouble4(b); }
  __forceinline vboold4 operator > (double          a, const vdouble4& b) { return vdouble4(a) >  b; }

  __forceinline vboold4 operator <=(const vdouble4& a, double          b) { return a <= vdouble4(b); }
  __forceinline vboold4 operator <=(double          a, const vdouble4& b) { return vdouble4(a) <= b; }

  __forceinline vboold4 eq(const vdouble4& a, const vdouble4& b) { return a == b; }
  __forceinline vboold4 ne(const vdouble4& a, const vdouble4& b) { return a != b; }
  __forceinline vboold4 lt(const vdouble4& a, const vdouble4& b) { return a <  b; }
  __forceinline vboold4 ge(const vdouble4& a, const vdouble4& b) { return a >= b; }
  __forceinline vboold4 gt(const vdouble4& a, const vdouble4& b) { return a >  b; }
  __forceinline vboold4 le(const vdouble4& a, const vdouble4& b) { return a <= b; }

#if defined(__AVX512VL__)
  __forceinline vboold4 eq(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_EQ); }
  __forceinline vboold4 ne(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_NE); }
  __forceinline vboold4 lt(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_LT); }
  __forceinline vboold4 ge(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_GE); }
  __forceinline vboold4 gt(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_GT); }
  __forceinline vboold4 le(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return _mm256_mask_cmp_pd_mask(mask, a, b, _MM_CMPINT_LE); }
#else
  __forceinline vboold4 eq(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a == b); }
  __forceinline vboold4 ne(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a != b); }
  __forceinline vboold4 lt(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a <  b); }
  __forceinline vboold4 ge(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a >= b); }
  __forceinline vboold4 gt(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a >  b); }
  __forceinline vboold4 le(const vboold4& mask, const vdouble4& a, const vdouble4& b) { return mask & (a <= b); }
#endif
 
  __forceinline vdouble4 select(const vboold4& m, const vdouble4& t, const vdouble4& f) {
#if defined(__AVX512VL__)
    return _mm256_mask_blend_pd(m, f, t);
#else
    return _mm256_blendv_pd(f, t, m);
#endif
  }

  __forceinline void xchg(const vboold4& m, vdouble4& a, vdouble4& b) {
    const vdouble4 c = a; a = select(m,b,a); b = select(m,c,b);
  }

  __forceinline vboold4 test(const vdouble4& a, const vdouble4& b) {
#if defined(__AVX512VL__)
    return _mm256_test_epi64_mask(_mm256_castpd_si256(a),_mm256_castpd_si256(b));
#else
    return _mm256_testz_si256(_mm256_castpd_si256(a),_mm256_castpd_si256(b));
#endif
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i0, int i1>
  __forceinline vdouble4 shuffle(const vdouble4& v) {
    return _mm256_permute_pd(v, (i1 << 3) | (i0 << 2) | (i1 << 1) | i0);
  }

  template<int i>
  __forceinline vdouble4 shuffle(const vdouble4& v) {
    return shuffle<i, i>(v);
  }

  template<int i0, int i1>
  __forceinline vdouble4 shuffle2(const vdouble4& v) {
    return _mm256_permute2f128_pd(v, v, (i1 << 4) | i0);
  }

  __forceinline double toScalar(const vdouble4& v) {
    return _mm_cvtsd_f64(_mm256_castpd256_pd128(v));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble4 vreduce_min2(const vdouble4& x) { return min(x, shuffle<1,0>(x)); }
  __forceinline vdouble4 vreduce_min (const vdouble4& y) { const vdouble4 x = vreduce_min2(y); return min(x, shuffle2<1,0>(x)); }

  __forceinline vdouble4 vreduce_max2(const vdouble4& x) { return max(x,shuffle<1,0>(x)); }
  __forceinline vdouble4 vreduce_max (const vdouble4& y) { const vdouble4 x = vreduce_max2(y); return max(x, shuffle2<1,0>(x)); }

  __forceinline vdouble4 vreduce_and2(const vdouble4& x) { return x & shuffle<1,0>(x); }
  __forceinline vdouble4 vreduce_and (const vdouble4& y) { const vdouble4 x = vreduce_and2(y); return x & shuffle2<1,0>(x); }

  __forceinline vdouble4 vreduce_or2(const vdouble4& x) { return x | shuffle<1,0>(x); }
  __forceinline vdouble4 vreduce_or (const vdouble4& y) { const vdouble4 x = vreduce_or2(y); return x | shuffle2<1,0>(x); }

  __forceinline vdouble4 vreduce_add2(const vdouble4& x) { return x + shuffle<1,0>(x); }
  __forceinline vdouble4 vreduce_add (const vdouble4& y) { const vdouble4 x = vreduce_add2(y); return x + shuffle2<1,0>(x); }

  __forceinline double reduce_add(const vdouble4& a) { return toScalar(vreduce_add(a)); }
  __forceinline double reduce_min(const vdouble4& a) { return toScalar(vreduce_min(a)); }
  __forceinline double reduce_max(const vdouble4& a) { return toScalar(vreduce_max(a)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////


  
  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline std::ostream& operator <<(std::ostream& cout, const vdouble4& v)
  {
    cout << "<" << v[0];
    for (size_t i=1; i<4; i++) cout << ", " << v[i];
    cout << ">";
    return cout;
  }
}
