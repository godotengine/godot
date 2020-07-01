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
  /* 8-wide AVX-512 64-bit double type */
  template<>
  struct vdouble<8>
  {
    typedef vboold8 Bool;

    enum  { size = 8 }; // number of SIMD elements
    union {              // data
      __m512d v;
      double i[8];
    };

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vdouble() {}
    __forceinline vdouble(const vdouble8& t) { v = t.v; }
    __forceinline vdouble8& operator =(const vdouble8& f) { v = f.v; return *this; }

    __forceinline vdouble(const __m512d& t) { v = t; }
    __forceinline operator __m512d() const { return v; }
    __forceinline operator __m256d() const { return _mm512_castpd512_pd256(v); }

    __forceinline vdouble(double i) {
      v = _mm512_set1_pd(i);
    }

    __forceinline vdouble(double a, double b, double c, double d) {
      v = _mm512_set4_pd(d,c,b,a);
    }

    __forceinline vdouble(double a0, double a1, double a2, double a3,
                          double a4, double a5, double a6, double a7)
    {
      v = _mm512_set_pd(a7,a6,a5,a4,a3,a2,a1,a0);
    }


    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vdouble(ZeroTy) : v(_mm512_setzero_pd()) {}
    __forceinline vdouble(OneTy)  : v(_mm512_set1_pd(1)) {}
    __forceinline vdouble(StepTy) : v(_mm512_set_pd(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0)) {}
    __forceinline vdouble(ReverseStepTy) : v(_mm512_setr_pd(7.0,6.0,5.0,4.0,3.0,2.0,1.0,0.0)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline void store_nt(void *__restrict__ ptr, const vdouble8& a) {
      _mm512_stream_pd((double*)ptr, a);
    }

    static __forceinline vdouble8 loadu(const void* addr) {
      return _mm512_loadu_pd((double*)addr);
    }

    static __forceinline vdouble8 load(const vdouble8* addr) {
      return _mm512_load_pd((double*)addr);
    }

    static __forceinline vdouble8 load(const double* addr) {
      return _mm512_load_pd(addr);
    }

    static __forceinline void store(void* ptr, const vdouble8& v) {
      _mm512_store_pd(ptr, v);
    }

    static __forceinline void storeu(void* ptr, const vdouble8& v) {
      _mm512_storeu_pd(ptr, v);
    }

    static __forceinline void storeu(const vboold8& mask, double* ptr, const vdouble8& f) {
      _mm512_mask_storeu_pd(ptr, mask, f);
    }

    static __forceinline void store(const vboold8& mask, void* addr, const vdouble8& v2) {
      _mm512_mask_store_pd(addr, mask, v2);
    }

    /* pass by value to avoid compiler generating inefficient code */
    static __forceinline void storeu_compact(const vboold8 mask,void * addr, const vdouble8& reg) {
      _mm512_mask_compressstoreu_pd(addr, mask, reg);
    }

    static __forceinline vdouble8 compact64bit(const vboold8& mask, vdouble8& v) {
      return _mm512_mask_compress_pd(v, mask, v);
    }

    static __forceinline vdouble8 compact(const vboold8& mask, vdouble8& v) {
      return _mm512_mask_compress_pd(v, mask, v);
    }

    static __forceinline vdouble8 compact(const vboold8& mask, const vdouble8& a, vdouble8& b) {
      return _mm512_mask_compress_pd(a, mask, b);
    }

    static __forceinline vdouble8 broadcast(const void* a) { return _mm512_set1_pd(*(double*)a); }

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline       double& operator [](size_t index)       { assert(index < 8); return i[index]; }
    __forceinline const double& operator [](size_t index) const { assert(index < 8); return i[index]; }

  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8 asDouble(const vllong8&  a) { return _mm512_castsi512_pd(a); }
  __forceinline vllong8  asLLong (const vdouble8& a) { return _mm512_castpd_si512(a); }

  __forceinline vdouble8 operator +(const vdouble8& a) { return a; }
  __forceinline vdouble8 operator -(const vdouble8& a) { return _mm512_sub_pd(_mm512_setzero_pd(), a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8 operator +(const vdouble8& a, const vdouble8& b) { return _mm512_add_pd(a, b); }
  __forceinline vdouble8 operator +(const vdouble8& a, double          b) { return a + vdouble8(b); }
  __forceinline vdouble8 operator +(double          a, const vdouble8& b) { return vdouble8(a) + b; }

  __forceinline vdouble8 operator -(const vdouble8& a, const vdouble8& b) { return _mm512_sub_pd(a, b); }
  __forceinline vdouble8 operator -(const vdouble8& a, double          b) { return a - vdouble8(b); }
  __forceinline vdouble8 operator -(double          a, const vdouble8& b) { return vdouble8(a) - b; }

  __forceinline vdouble8 operator *(const vdouble8& a, const vdouble8& b) { return _mm512_mul_pd(a, b); }
  __forceinline vdouble8 operator *(const vdouble8& a, double          b) { return a * vdouble8(b); }
  __forceinline vdouble8 operator *(double          a, const vdouble8& b) { return vdouble8(a) * b; }

  __forceinline vdouble8 operator &(const vdouble8& a, const vdouble8& b) { return _mm512_and_pd(a, b); }
  __forceinline vdouble8 operator &(const vdouble8& a, double          b) { return a & vdouble8(b); }
  __forceinline vdouble8 operator &(double          a, const vdouble8& b) { return vdouble8(a) & b; }

  __forceinline vdouble8 operator |(const vdouble8& a, const vdouble8& b) { return _mm512_or_pd(a, b); }
  __forceinline vdouble8 operator |(const vdouble8& a, double          b) { return a | vdouble8(b); }
  __forceinline vdouble8 operator |(double          a, const vdouble8& b) { return vdouble8(a) | b; }

  __forceinline vdouble8 operator ^(const vdouble8& a, const vdouble8& b) { return _mm512_xor_pd(a, b); }
  __forceinline vdouble8 operator ^(const vdouble8& a, double          b) { return a ^ vdouble8(b); }
  __forceinline vdouble8 operator ^(double          a, const vdouble8& b) { return vdouble8(a) ^ b; }

  __forceinline vdouble8 operator <<(const vdouble8& a, const unsigned int n) { return _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_castpd_si512(a), n)); }
  __forceinline vdouble8 operator >>(const vdouble8& a, const unsigned int n) { return _mm512_castsi512_pd(_mm512_srai_epi64(_mm512_castpd_si512(a), n)); }

  __forceinline vdouble8 operator <<(const vdouble8& a, const vllong8& n) { return _mm512_castsi512_pd(_mm512_sllv_epi64(_mm512_castpd_si512(a), n)); }
  __forceinline vdouble8 operator >>(const vdouble8& a, const vllong8& n) { return _mm512_castsi512_pd(_mm512_srav_epi64(_mm512_castpd_si512(a), n)); }

  __forceinline vdouble8 sll (const vdouble8& a, const unsigned int b) { return  _mm512_castsi512_pd(_mm512_slli_epi64(_mm512_castpd_si512(a), b)); }
  __forceinline vdouble8 sra (const vdouble8& a, const unsigned int b) { return  _mm512_castsi512_pd(_mm512_srai_epi64(_mm512_castpd_si512(a), b)); }
  __forceinline vdouble8 srl (const vdouble8& a, const unsigned int b) { return  _mm512_castsi512_pd(_mm512_srli_epi64(_mm512_castpd_si512(a), b)); }

  __forceinline vdouble8 min(const vdouble8& a, const vdouble8& b) { return _mm512_min_pd(a, b); }
  __forceinline vdouble8 min(const vdouble8& a, double          b) { return min(a,vdouble8(b)); }
  __forceinline vdouble8 min(double          a, const vdouble8& b) { return min(vdouble8(a),b); }

  __forceinline vdouble8 max(const vdouble8& a, const vdouble8& b) { return _mm512_max_pd(a, b); }
  __forceinline vdouble8 max(const vdouble8& a, double          b) { return max(a,vdouble8(b)); }
  __forceinline vdouble8 max(double          a, const vdouble8& b) { return max(vdouble8(a),b); }

  __forceinline vdouble8 mask_add(const vboold8& mask, vdouble8& c, const vdouble8& a, const vdouble8& b) { return _mm512_mask_add_pd(c,mask,a,b); }
  __forceinline vdouble8 mask_sub(const vboold8& mask, vdouble8& c, const vdouble8& a, const vdouble8& b) { return _mm512_mask_sub_pd(c,mask,a,b); }

  __forceinline vdouble8 mask_and(const vboold8& m,vdouble8& c, const vdouble8& a, const vdouble8& b) { return _mm512_mask_and_pd(c,m,a,b); }
  __forceinline vdouble8 mask_or (const vboold8& m,vdouble8& c, const vdouble8& a, const vdouble8& b) { return _mm512_mask_or_pd(c,m,a,b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Ternary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8 madd (const vdouble8& a, const vdouble8& b, const vdouble8& c) { return _mm512_fmadd_pd(a,b,c); }
  __forceinline vdouble8 msub (const vdouble8& a, const vdouble8& b, const vdouble8& c) { return _mm512_fmsub_pd(a,b,c); }
  __forceinline vdouble8 nmadd(const vdouble8& a, const vdouble8& b, const vdouble8& c) { return _mm512_fnmadd_pd(a,b,c); }
  __forceinline vdouble8 nmsub(const vdouble8& a, const vdouble8& b, const vdouble8& c) { return _mm512_fnmsub_pd(a,b,c); }


  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8& operator +=(vdouble8& a, const vdouble8& b) { return a = a + b; }
  __forceinline vdouble8& operator +=(vdouble8& a, double          b) { return a = a + b; }

  __forceinline vdouble8& operator -=(vdouble8& a, const vdouble8& b) { return a = a - b; }
  __forceinline vdouble8& operator -=(vdouble8& a, double          b) { return a = a - b; }

  __forceinline vdouble8& operator *=(vdouble8& a, const vdouble8& b) { return a = a * b; }
  __forceinline vdouble8& operator *=(vdouble8& a, double          b) { return a = a * b; }

  __forceinline vdouble8& operator &=(vdouble8& a, const vdouble8& b) { return a = a & b; }
  __forceinline vdouble8& operator &=(vdouble8& a, double          b) { return a = a & b; }

  __forceinline vdouble8& operator |=(vdouble8& a, const vdouble8& b) { return a = a | b; }
  __forceinline vdouble8& operator |=(vdouble8& a, double          b) { return a = a | b; }

  __forceinline vdouble8& operator <<=(vdouble8& a, const double b) { return a = a << b; }
  __forceinline vdouble8& operator >>=(vdouble8& a, const double b) { return a = a >> b; }


  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboold8 operator ==(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboold8 operator ==(const vdouble8& a, double          b) { return a == vdouble8(b); }
  __forceinline vboold8 operator ==(double          a, const vdouble8& b) { return vdouble8(a) == b; }

  __forceinline vboold8 operator !=(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboold8 operator !=(const vdouble8& a, double          b) { return a != vdouble8(b); }
  __forceinline vboold8 operator !=(double          a, const vdouble8& b) { return vdouble8(a) != b; }

  __forceinline vboold8 operator < (const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboold8 operator < (const vdouble8& a, double          b) { return a <  vdouble8(b); }
  __forceinline vboold8 operator < (double          a, const vdouble8& b) { return vdouble8(a) <  b; }

  __forceinline vboold8 operator >=(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboold8 operator >=(const vdouble8& a, double          b) { return a >= vdouble8(b); }
  __forceinline vboold8 operator >=(double          a, const vdouble8& b) { return vdouble8(a) >= b; }

  __forceinline vboold8 operator > (const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboold8 operator > (const vdouble8& a, double          b) { return a >  vdouble8(b); }
  __forceinline vboold8 operator > (double          a, const vdouble8& b) { return vdouble8(a) >  b; }

  __forceinline vboold8 operator <=(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_LE); }
  __forceinline vboold8 operator <=(const vdouble8& a, double          b) { return a <= vdouble8(b); }
  __forceinline vboold8 operator <=(double          a, const vdouble8& b) { return vdouble8(a) <= b; }

  __forceinline vboold8 eq(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_EQ); }
  __forceinline vboold8 ne(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_NE); }
  __forceinline vboold8 lt(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_LT); }
  __forceinline vboold8 ge(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_GE); }
  __forceinline vboold8 gt(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_GT); }
  __forceinline vboold8 le(const vdouble8& a, const vdouble8& b) { return _mm512_cmp_pd_mask(a,b,_MM_CMPINT_LE); }

  __forceinline vboold8 eq(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_EQ); }
  __forceinline vboold8 ne(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_NE); }
  __forceinline vboold8 lt(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_LT); }
  __forceinline vboold8 ge(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_GE); }
  __forceinline vboold8 gt(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_GT); }
  __forceinline vboold8 le(const vboold8 mask, const vdouble8& a, const vdouble8& b) { return _mm512_mask_cmp_pd_mask(mask,a,b,_MM_CMPINT_LE); }

  __forceinline vdouble8 select(const vboold8& m, const vdouble8& t, const vdouble8& f) {
    return _mm512_mask_or_pd(f,m,t,t);
  }

  __forceinline void xchg(const vboold8& m, vdouble8& a, vdouble8& b) {
    const vdouble8 c = a; a = select(m,b,a); b = select(m,c,b);
  }

  __forceinline vboold8 test(const vboold8& m, const vdouble8& a, const vdouble8& b) {
    return _mm512_mask_test_epi64_mask(m,_mm512_castpd_si512(a),_mm512_castpd_si512(b));
  }

  __forceinline vboold8 test(const vdouble8& a, const vdouble8& b) {
    return _mm512_test_epi64_mask(_mm512_castpd_si512(a),_mm512_castpd_si512(b));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  template<int i0, int i1>
  __forceinline vdouble8 shuffle(const vdouble8& v) {
    return _mm512_permute_pd(v, (i1 << 7) | (i0 << 6) | (i1 << 5) | (i0 << 4) | (i1 << 3) | (i0 << 2) | (i1 << 1) | i0);
  }

  template<int i>
  __forceinline vdouble8 shuffle(const vdouble8& v) {
    return shuffle<i, i>(v);
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vdouble8 shuffle(const vdouble8& v) {
    return _mm512_permutex_pd(v, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<int i0, int i1>
  __forceinline vdouble8 shuffle4(const vdouble8& v) {
    return _mm512_shuffle_f64x2(v, v, _MM_SHUFFLE(i1*2+1, i1*2, i0*2+1, i0*2));
  }

  template<int i>
  __forceinline vdouble8 shuffle4(const vdouble8& v) {
    return shuffle4<i, i>(v);
  }
  
  template<int i>
  __forceinline vdouble8 align_shift_right(const vdouble8& a, const vdouble8& b) {
    return _mm512_castsi512_pd(_mm512_alignr_epi64(_mm512_castpd_si512(a), _mm512_castpd_si512(b), i));
  }

  __forceinline double toScalar(const vdouble8& v) {
    return _mm_cvtsd_f64(_mm512_castpd512_pd128(v));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8 vreduce_add2(vdouble8 x) {                      return x + shuffle<1,0,3,2>(x); }
  __forceinline vdouble8 vreduce_add4(vdouble8 x) { x = vreduce_add2(x); return x + shuffle<2,3,0,1>(x); }
  __forceinline vdouble8 vreduce_add (vdouble8 x) { x = vreduce_add4(x); return x + shuffle4<1,0>(x); }

  __forceinline vdouble8 vreduce_min2(vdouble8 x) {                      return min(x, shuffle<1,0,3,2>(x)); }
  __forceinline vdouble8 vreduce_min4(vdouble8 x) { x = vreduce_min2(x); return min(x, shuffle<2,3,0,1>(x)); }
  __forceinline vdouble8 vreduce_min (vdouble8 x) { x = vreduce_min4(x); return min(x, shuffle4<1,0>(x)); }

  __forceinline vdouble8 vreduce_max2(vdouble8 x) {                      return max(x, shuffle<1,0,3,2>(x)); }
  __forceinline vdouble8 vreduce_max4(vdouble8 x) { x = vreduce_max2(x); return max(x, shuffle<2,3,0,1>(x)); }
  __forceinline vdouble8 vreduce_max (vdouble8 x) { x = vreduce_max4(x); return max(x, shuffle4<1,0>(x)); }

  __forceinline double reduce_add(const vdouble8& v) { return toScalar(vreduce_add(v)); }
  __forceinline double reduce_min(const vdouble8& v) { return toScalar(vreduce_min(v)); }
  __forceinline double reduce_max(const vdouble8& v) { return toScalar(vreduce_max(v)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Memory load and store operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vdouble8 permute(const vdouble8& v, const vllong8& index) {
    return _mm512_permutexvar_pd(index, v);
  }

  __forceinline vdouble8 reverse(const vdouble8& a) {
    return permute(a, vllong8(reverse_step));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline std::ostream& operator <<(std::ostream& cout, const vdouble8& v)
  {
    cout << "<" << v[0];
    for (size_t i=1; i<8; i++) cout << ", " << v[i];
    cout << ">";
    return cout;
  }
}
