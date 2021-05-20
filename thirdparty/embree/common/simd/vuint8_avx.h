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
  struct vuint<8>
  {
    ALIGNED_STRUCT_(32);   

    typedef vboolf8 Bool;
    typedef vuint8   Int;
    typedef vfloat8 Float;

    enum  { size = 8 };        // number of SIMD elements
    union {                    // data
      __m256i v;
      struct { __m128i vl,vh; };
      unsigned int i[8];
    }; 

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vuint() {}
    __forceinline vuint(const vuint8& a) { v = a.v; }
    __forceinline vuint8& operator =(const vuint8& a) { v = a.v; return *this; }

    __forceinline vuint(__m256i a) : v(a) {}
    __forceinline operator const __m256i&() const { return v; }
    __forceinline operator       __m256i&()       { return v; }

    __forceinline explicit vuint(const vuint4& a) : v(_mm256_insertf128_si256(_mm256_castsi128_si256(a),a,1)) {}
    __forceinline vuint(const vuint4& a, const vuint4& b) : v(_mm256_insertf128_si256(_mm256_castsi128_si256(a),b,1)) {}
    __forceinline vuint(const __m128i& a, const __m128i& b) : vl(a), vh(b) {}
 
    __forceinline explicit vuint(const unsigned int* a) : v(_mm256_castps_si256(_mm256_loadu_ps((const float*)a))) {}
    __forceinline vuint(unsigned int a) : v(_mm256_set1_epi32(a)) {}
    __forceinline vuint(unsigned int a, unsigned int b) : v(_mm256_set_epi32(b, a, b, a, b, a, b, a)) {}
    __forceinline vuint(unsigned int a, unsigned int b, unsigned int c, unsigned int d) : v(_mm256_set_epi32(d, c, b, a, d, c, b, a)) {}
    __forceinline vuint(unsigned int a, unsigned int b, unsigned int c, unsigned int d, unsigned int e, unsigned int f, unsigned int g, unsigned int vh) : v(_mm256_set_epi32(vh, g, f, e, d, c, b, a)) {}

    __forceinline explicit vuint(__m256 a) : v(_mm256_cvtps_epi32(a)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vuint(ZeroTy)   : v(_mm256_setzero_si256()) {}
    __forceinline vuint(OneTy)    : v(_mm256_set1_epi32(1)) {}
    __forceinline vuint(PosInfTy) : v(_mm256_set1_epi32(0xFFFFFFFF)) {}
    __forceinline vuint(StepTy)   : v(_mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0)) {}
    __forceinline vuint(UndefinedTy) : v(_mm256_undefined_si256()) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Loads and Stores
    ////////////////////////////////////////////////////////////////////////////////

    static __forceinline vuint8 load (const void* a) { return _mm256_castps_si256(_mm256_load_ps((float*)a)); }
    static __forceinline vuint8 loadu(const void* a) { return _mm256_castps_si256(_mm256_loadu_ps((float*)a)); }

    static __forceinline vuint8 load (const vboolf8& mask, const void* a) { return _mm256_castps_si256(_mm256_maskload_ps((float*)a,mask)); }
    static __forceinline vuint8 loadu(const vboolf8& mask, const void* a) { return _mm256_castps_si256(_mm256_maskload_ps((float*)a,mask)); }

    static __forceinline void store (void* ptr, const vuint8& f) { _mm256_store_ps((float*)ptr,_mm256_castsi256_ps(f)); }
    static __forceinline void storeu(void* ptr, const vuint8& f) { _mm256_storeu_ps((float*)ptr,_mm256_castsi256_ps(f)); }
    
    static __forceinline void store (const vboolf8& mask, void* ptr, const vuint8& f) { _mm256_maskstore_ps((float*)ptr,(__m256i)mask,_mm256_castsi256_ps(f)); }
    static __forceinline void storeu(const vboolf8& mask, void* ptr, const vuint8& f) { _mm256_maskstore_ps((float*)ptr,(__m256i)mask,_mm256_castsi256_ps(f)); }

    static __forceinline void store_nt(void* ptr, const vuint8& v) {
      _mm256_stream_ps((float*)ptr,_mm256_castsi256_ps(v));
    }

    static __forceinline vuint8 load(const unsigned char* ptr) {
      vuint4 il = vuint4::load(ptr+0);
      vuint4 ih = vuint4::load(ptr+4);
      return vuint8(il,ih);
    }

    static __forceinline vuint8 loadu(const unsigned char* ptr) {
      vuint4 il = vuint4::loadu(ptr+0);
      vuint4 ih = vuint4::loadu(ptr+4);
      return vuint8(il,ih);
    }

    static __forceinline vuint8 load(const unsigned short* ptr) {
      vuint4 il = vuint4::load(ptr+0);
      vuint4 ih = vuint4::load(ptr+4);
      return vuint8(il,ih);
    }

    static __forceinline vuint8 loadu(const unsigned short* ptr) {
      vuint4 il = vuint4::loadu(ptr+0);
      vuint4 ih = vuint4::loadu(ptr+4);
      return vuint8(il,ih);
    }

    static __forceinline void store(unsigned char* ptr, const vuint8& i) {
      vuint4 il(i.vl);
      vuint4 ih(i.vh);
      vuint4::store(ptr + 0,il);
      vuint4::store(ptr + 4,ih);
    }

    static __forceinline void store(unsigned short* ptr, const vuint8& v) {
      for (size_t i=0;i<8;i++)
        ptr[i] = (unsigned short)v[i];
    }

    template<int scale = 4>
    static __forceinline vuint8 gather(const unsigned int* ptr, const vint8& index) {
      return vuint8(
          *(unsigned int*)(((char*)ptr)+scale*index[0]),
          *(unsigned int*)(((char*)ptr)+scale*index[1]),
          *(unsigned int*)(((char*)ptr)+scale*index[2]),
          *(unsigned int*)(((char*)ptr)+scale*index[3]),
          *(unsigned int*)(((char*)ptr)+scale*index[4]),
          *(unsigned int*)(((char*)ptr)+scale*index[5]),
          *(unsigned int*)(((char*)ptr)+scale*index[6]),
          *(unsigned int*)(((char*)ptr)+scale*index[7]));
    }

    template<int scale = 4>
    static __forceinline vuint8 gather(const vboolf8& mask, const unsigned int* ptr, const vint8& index) {
      vuint8 r = zero;
      if (likely(mask[0])) r[0] = *(unsigned int*)(((char*)ptr)+scale*index[0]);
      if (likely(mask[1])) r[1] = *(unsigned int*)(((char*)ptr)+scale*index[1]);
      if (likely(mask[2])) r[2] = *(unsigned int*)(((char*)ptr)+scale*index[2]);
      if (likely(mask[3])) r[3] = *(unsigned int*)(((char*)ptr)+scale*index[3]);
      if (likely(mask[4])) r[4] = *(unsigned int*)(((char*)ptr)+scale*index[4]);
      if (likely(mask[5])) r[5] = *(unsigned int*)(((char*)ptr)+scale*index[5]);
      if (likely(mask[6])) r[6] = *(unsigned int*)(((char*)ptr)+scale*index[6]);
      if (likely(mask[7])) r[7] = *(unsigned int*)(((char*)ptr)+scale*index[7]);
      return r;
    }

    template<int scale = 4>
    static __forceinline void scatter(void* ptr, const vint8& ofs, const vuint8& v)
    {
      *(unsigned int*)(((char*)ptr)+scale*ofs[0]) = v[0];
      *(unsigned int*)(((char*)ptr)+scale*ofs[1]) = v[1];
      *(unsigned int*)(((char*)ptr)+scale*ofs[2]) = v[2];
      *(unsigned int*)(((char*)ptr)+scale*ofs[3]) = v[3];
      *(unsigned int*)(((char*)ptr)+scale*ofs[4]) = v[4];
      *(unsigned int*)(((char*)ptr)+scale*ofs[5]) = v[5];
      *(unsigned int*)(((char*)ptr)+scale*ofs[6]) = v[6];
      *(unsigned int*)(((char*)ptr)+scale*ofs[7]) = v[7];
    }

    template<int scale = 4>
    static __forceinline void scatter(const vboolf8& mask, void* ptr, const vint8& ofs, const vuint8& v)
    {
      if (likely(mask[0])) *(unsigned int*)(((char*)ptr)+scale*ofs[0]) = v[0];
      if (likely(mask[1])) *(unsigned int*)(((char*)ptr)+scale*ofs[1]) = v[1];
      if (likely(mask[2])) *(unsigned int*)(((char*)ptr)+scale*ofs[2]) = v[2];
      if (likely(mask[3])) *(unsigned int*)(((char*)ptr)+scale*ofs[3]) = v[3];
      if (likely(mask[4])) *(unsigned int*)(((char*)ptr)+scale*ofs[4]) = v[4];
      if (likely(mask[5])) *(unsigned int*)(((char*)ptr)+scale*ofs[5]) = v[5];
      if (likely(mask[6])) *(unsigned int*)(((char*)ptr)+scale*ofs[6]) = v[6];
      if (likely(mask[7])) *(unsigned int*)(((char*)ptr)+scale*ofs[7]) = v[7];
    }


    static __forceinline vuint8 broadcast64(const long long& a) { return _mm256_set1_epi64x(a); }
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline const unsigned int& operator [](size_t index) const { assert(index < 8); return i[index]; }
    __forceinline       unsigned int& operator [](size_t index)       { assert(index < 8); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 asBool(const vuint8& a) { return _mm256_castsi256_ps(a); }

  __forceinline vuint8 operator +(const vuint8& a) { return a; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint8 operator +(const vuint8& a, const vuint8& b) { return vuint8(_mm_add_epi32(a.vl, b.vl), _mm_add_epi32(a.vh, b.vh)); }
  __forceinline vuint8 operator +(const vuint8& a, unsigned int          b) { return a + vuint8(b); }
  __forceinline vuint8 operator +(unsigned int          a, const vuint8& b) { return vuint8(a) + b; }

  __forceinline vuint8 operator -(const vuint8& a, const vuint8& b) { return vuint8(_mm_sub_epi32(a.vl, b.vl), _mm_sub_epi32(a.vh, b.vh)); }
  __forceinline vuint8 operator -(const vuint8& a, unsigned int          b) { return a - vuint8(b); }
  __forceinline vuint8 operator -(unsigned int          a, const vuint8& b) { return vuint8(a) - b; }

  //__forceinline vuint8 operator *(const vuint8& a, const vuint8& b) { return vuint8(_mm_mullo_epu32(a.vl, b.vl), _mm_mullo_epu32(a.vh, b.vh)); }
  //__forceinline vuint8 operator *(const vuint8& a, unsigned int          b) { return a * vuint8(b); }
  //__forceinline vuint8 operator *(unsigned int          a, const vuint8& b) { return vuint8(a) * b; }

  __forceinline vuint8 operator &(const vuint8& a, const vuint8& b) { return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b))); }
  __forceinline vuint8 operator &(const vuint8& a, unsigned int          b) { return a & vuint8(b); }
  __forceinline vuint8 operator &(unsigned int          a, const vuint8& b) { return vuint8(a) & b; }

  __forceinline vuint8 operator |(const vuint8& a, const vuint8& b) { return _mm256_castps_si256(_mm256_or_ps (_mm256_castsi256_ps(a), _mm256_castsi256_ps(b))); }
  __forceinline vuint8 operator |(const vuint8& a, unsigned int          b) { return a | vuint8(b); }
  __forceinline vuint8 operator |(unsigned int          a, const vuint8& b) { return vuint8(a) | b; }

  __forceinline vuint8 operator ^(const vuint8& a, const vuint8& b) { return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b))); }
  __forceinline vuint8 operator ^(const vuint8& a, unsigned int          b) { return a ^ vuint8(b); }
  __forceinline vuint8 operator ^(unsigned int          a, const vuint8& b) { return vuint8(a) ^ b; }

  __forceinline vuint8 operator <<(const vuint8& a, unsigned int n) { return vuint8(_mm_slli_epi32(a.vl, n), _mm_slli_epi32(a.vh, n)); }
  __forceinline vuint8 operator >>(const vuint8& a, unsigned int n) { return vuint8(_mm_srai_epi32(a.vl, n), _mm_srli_epi32(a.vh, n)); }

  __forceinline vuint8 sll (const vuint8& a, unsigned int b) { return vuint8(_mm_slli_epi32(a.vl, b), _mm_slli_epi32(a.vh, b)); }
  __forceinline vuint8 sra (const vuint8& a, unsigned int b) { return vuint8(_mm_srai_epi32(a.vl, b), _mm_srai_epi32(a.vh, b)); }
  __forceinline vuint8 srl (const vuint8& a, unsigned int b) { return vuint8(_mm_srli_epi32(a.vl, b), _mm_srli_epi32(a.vh, b)); }
  
  __forceinline vuint8 min(const vuint8& a, const vuint8& b) { return vuint8(_mm_min_epu32(a.vl, b.vl), _mm_min_epu32(a.vh, b.vh)); }
  __forceinline vuint8 min(const vuint8& a, unsigned int          b) { return min(a,vuint8(b)); }
  __forceinline vuint8 min(unsigned int          a, const vuint8& b) { return min(vuint8(a),b); }

  __forceinline vuint8 max(const vuint8& a, const vuint8& b) { return vuint8(_mm_max_epu32(a.vl, b.vl), _mm_max_epu32(a.vh, b.vh)); }
  __forceinline vuint8 max(const vuint8& a, unsigned int          b) { return max(a,vuint8(b)); }
  __forceinline vuint8 max(unsigned int          a, const vuint8& b) { return max(vuint8(a),b); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint8& operator +=(vuint8& a, const vuint8& b) { return a = a + b; }
  __forceinline vuint8& operator +=(vuint8& a, unsigned int          b) { return a = a + b; }
  
  __forceinline vuint8& operator -=(vuint8& a, const vuint8& b) { return a = a - b; }
  __forceinline vuint8& operator -=(vuint8& a, unsigned int          b) { return a = a - b; }
  
  //__forceinline vuint8& operator *=(vuint8& a, const vuint8& b) { return a = a * b; }
  //__forceinline vuint8& operator *=(vuint8& a, unsigned int          b) { return a = a * b; }
  
  __forceinline vuint8& operator &=(vuint8& a, const vuint8& b) { return a = a & b; }
  __forceinline vuint8& operator &=(vuint8& a, unsigned int          b) { return a = a & b; }
  
  __forceinline vuint8& operator |=(vuint8& a, const vuint8& b) { return a = a | b; }
  __forceinline vuint8& operator |=(vuint8& a, unsigned int          b) { return a = a | b; }
  
  __forceinline vuint8& operator <<=(vuint8& a, unsigned int b) { return a = a << b; }
  __forceinline vuint8& operator >>=(vuint8& a, unsigned int b) { return a = a >> b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator ==(const vuint8& a, const vuint8& b) { return vboolf8(_mm_castsi128_ps(_mm_cmpeq_epi32 (a.vl, b.vl)),
                                                                                       _mm_castsi128_ps(_mm_cmpeq_epi32 (a.vh, b.vh))); }
  __forceinline vboolf8 operator ==(const vuint8& a, unsigned int          b) { return a == vuint8(b); }
  __forceinline vboolf8 operator ==(unsigned int          a, const vuint8& b) { return vuint8(a) == b; }
  
  __forceinline vboolf8 operator !=(const vuint8& a, const vuint8& b) { return !(a == b); }
  __forceinline vboolf8 operator !=(const vuint8& a, unsigned int          b) { return a != vuint8(b); }
  __forceinline vboolf8 operator !=(unsigned int          a, const vuint8& b) { return vuint8(a) != b; }
  
  //__forceinline vboolf8 operator < (const vuint8& a, const vuint8& b) { return vboolf8(_mm_castsi128_ps(_mm_cmplt_epu32 (a.vl, b.vl)),
  //                                                                                     _mm_castsi128_ps(_mm_cmplt_epu32 (a.vh, b.vh))); }
  //__forceinline vboolf8 operator < (const vuint8& a, unsigned int          b) { return a <  vuint8(b); }
  //__forceinline vboolf8 operator < (unsigned int          a, const vuint8& b) { return vuint8(a) <  b; }
  
  //__forceinline vboolf8 operator >=(const vuint8& a, const vuint8& b) { return !(a <  b); }
  //__forceinline vboolf8 operator >=(const vuint8& a, unsigned int          b) { return a >= vuint8(b); }
  //__forceinline vboolf8 operator >=(unsigned int          a, const vuint8& b) { return vuint8(a) >= b; }

  //__forceinline vboolf8 operator > (const vuint8& a, const vuint8& b) { return vboolf8(_mm_castsi128_ps(_mm_cmpgt_epu32 (a.vl, b.vl)),
  //                                                                                     _mm_castsi128_ps(_mm_cmpgt_epu32 (a.vh, b.vh))); }
  //__forceinline vboolf8 operator > (const vuint8& a, unsigned int          b) { return a >  vuint8(b); }
  //__forceinline vboolf8 operator > (unsigned int          a, const vuint8& b) { return vuint8(a) >  b; }

  //__forceinline vboolf8 operator <=(const vuint8& a, const vuint8& b) { return !(a >  b); }
  //__forceinline vboolf8 operator <=(const vuint8& a, unsigned int          b) { return a <= vuint8(b); }
  //__forceinline vboolf8 operator <=(unsigned int          a, const vuint8& b) { return vuint8(a) <= b; }

  __forceinline vboolf8 eq(const vuint8& a, const vuint8& b) { return a == b; }
  __forceinline vboolf8 ne(const vuint8& a, const vuint8& b) { return a != b; }

  __forceinline vboolf8 eq(const vboolf8& mask, const vuint8& a, const vuint8& b) { return mask & (a == b); }
  __forceinline vboolf8 ne(const vboolf8& mask, const vuint8& a, const vuint8& b) { return mask & (a != b); }

  __forceinline vuint8 select(const vboolf8& m, const vuint8& t, const vuint8& f) {
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(f), _mm256_castsi256_ps(t), m)); 
  }


  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vuint8 unpacklo(const vuint8& a, const vuint8& b) { return _mm256_castps_si256(_mm256_unpacklo_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b))); }
  __forceinline vuint8 unpackhi(const vuint8& a, const vuint8& b) { return _mm256_castps_si256(_mm256_unpackhi_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b))); }

  template<int i>
  __forceinline vuint8 shuffle(const vuint8& v) {
    return _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(v), _MM_SHUFFLE(i, i, i, i)));
  }

  template<int i0, int i1>
  __forceinline vuint8 shuffle4(const vuint8& v) {
    return _mm256_permute2f128_si256(v, v, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1>
  __forceinline vuint8 shuffle4(const vuint8& a, const vuint8& b) {
    return _mm256_permute2f128_si256(a, b, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vuint8 shuffle(const vuint8& v) {
    return _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(v), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vuint8 shuffle(const vuint8& a, const vuint8& b) {
    return _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), _MM_SHUFFLE(i3, i2, i1, i0)));
  }

  template<> __forceinline vuint8 shuffle<0, 0, 2, 2>(const vuint8& v) { return _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(v))); }
  template<> __forceinline vuint8 shuffle<1, 1, 3, 3>(const vuint8& v) { return _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(v))); }
  template<> __forceinline vuint8 shuffle<0, 1, 0, 1>(const vuint8& v) { return _mm256_castps_si256(_mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(_mm256_castsi256_ps(v))))); }

  template<int i> __forceinline vuint8 insert4(const vuint8& a, const vuint4& b) { return _mm256_insertf128_si256(a, b, i); }
  template<int i> __forceinline vuint4 extract4(const vuint8& a) { return _mm256_extractf128_si256(a, i); }
  template<> __forceinline vuint4 extract4<0>(const vuint8& a) { return _mm256_castsi256_si128(a); }

  __forceinline int toScalar(const vuint8& v) { return _mm_cvtsi128_si32(_mm256_castsi256_si128(v)); }


  ////////////////////////////////////////////////////////////////////////////////
  /// Reductions
  ////////////////////////////////////////////////////////////////////////////////

  //__forceinline vuint8 vreduce_min2(const vuint8& v) { return min(v,shuffle<1,0,3,2>(v)); }
  //__forceinline vuint8 vreduce_min4(const vuint8& v) { vuint8 v1 = vreduce_min2(v); return min(v1,shuffle<2,3,0,1>(v1)); }
  //__forceinline vuint8 vreduce_min (const vuint8& v) { vuint8 v1 = vreduce_min4(v); return min(v1,shuffle4<1,0>(v1)); }

  //__forceinline vuint8 vreduce_max2(const vuint8& v) { return max(v,shuffle<1,0,3,2>(v)); }
  //__forceinline vuint8 vreduce_max4(const vuint8& v) { vuint8 v1 = vreduce_max2(v); return max(v1,shuffle<2,3,0,1>(v1)); }
  //__forceinline vuint8 vreduce_max (const vuint8& v) { vuint8 v1 = vreduce_max4(v); return max(v1,shuffle4<1,0>(v1)); }

  __forceinline vuint8 vreduce_add2(const vuint8& v) { return v + shuffle<1,0,3,2>(v); }
  __forceinline vuint8 vreduce_add4(const vuint8& v) { vuint8 v1 = vreduce_add2(v); return v1 + shuffle<2,3,0,1>(v1); }
  __forceinline vuint8 vreduce_add (const vuint8& v) { vuint8 v1 = vreduce_add4(v); return v1 + shuffle4<1,0>(v1); }

  //__forceinline int reduce_min(const vuint8& v) { return toScalar(vreduce_min(v)); }
  //__forceinline int reduce_max(const vuint8& v) { return toScalar(vreduce_max(v)); }
  __forceinline int reduce_add(const vuint8& v) { return toScalar(vreduce_add(v)); }

  //__forceinline size_t select_min(const vuint8& v) { return bsf(movemask(v == vreduce_min(v))); }
  //__forceinline size_t select_max(const vuint8& v) { return bsf(movemask(v == vreduce_max(v))); }

  //__forceinline size_t select_min(const vboolf8& valid, const vuint8& v) { const vuint8 a = select(valid,v,vuint8(pos_inf)); return bsf(movemask(valid & (a == vreduce_min(a)))); }
  //__forceinline size_t select_max(const vboolf8& valid, const vuint8& v) { const vuint8 a = select(valid,v,vuint8(neg_inf)); return bsf(movemask(valid & (a == vreduce_max(a)))); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator <<(embree_ostream cout, const vuint8& a) {
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
