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
  /* 8-wide AVX bool type */
  template<>
  struct vboolf<8>
  {
    typedef vboolf8 Bool;
    typedef vint8   Int;
    typedef vfloat8 Float;

    enum  { size = 8 };       // number of SIMD elements
    union {                   // data
      __m256 v;
      struct { __m128 vl,vh; };
      int i[8];
    };  

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf() {}
    __forceinline vboolf(const vboolf8& a) { v = a.v; }
    __forceinline vboolf8& operator =(const vboolf8& a) { v = a.v; return *this; }

    __forceinline vboolf(__m256 a) : v(a) {}
    __forceinline operator const __m256&() const { return v; }
    __forceinline operator const __m256i() const { return _mm256_castps_si256(v); }
    __forceinline operator const __m256d() const { return _mm256_castps_pd(v); }

    __forceinline vboolf(int a)
    {
      assert(a >= 0 && a <= 255);
#if defined (__AVX2__)
      const __m256i mask = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1);
      const __m256i b = _mm256_set1_epi32(a);
      const __m256i c = _mm256_and_si256(b,mask);
      v = _mm256_castsi256_ps(_mm256_cmpeq_epi32(c,mask));
#else
      vl = mm_lookupmask_ps[a & 0xF];
      vh = mm_lookupmask_ps[a >> 4];
#endif
    }

    __forceinline vboolf(const vboolf4& a) : v(_mm256_insertf128_ps(_mm256_castps128_ps256(a),a,1)) {}
    __forceinline vboolf(const vboolf4& a, const vboolf4& b) : v(_mm256_insertf128_ps(_mm256_castps128_ps256(a),b,1)) {}
    __forceinline vboolf(__m128 a, __m128 b) : vl(a), vh(b) {}

    __forceinline vboolf(bool a) : v(vboolf8(vboolf4(a), vboolf4(a))) {}
    __forceinline vboolf(bool a, bool b) : v(vboolf8(vboolf4(a), vboolf4(b))) {}
    __forceinline vboolf(bool a, bool b, bool c, bool d) : v(vboolf8(vboolf4(a,b), vboolf4(c,d))) {}
    __forceinline vboolf(bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h) : v(vboolf8(vboolf4(a,b,c,d), vboolf4(e,f,g,h))) {}

    /* return int32 mask */
    __forceinline __m256i mask32() const { 
      return _mm256_castps_si256(v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf(FalseTy) : v(_mm256_setzero_ps()) {}
    __forceinline vboolf(TrueTy)  : v(_mm256_cmp_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _CMP_EQ_OQ)) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool operator [](size_t index) const { assert(index < 8); return (_mm256_movemask_ps(v) >> index) & 1; }
    __forceinline int& operator [](size_t index)       { assert(index < 8); return i[index]; }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator !(const vboolf8& a) { return _mm256_xor_ps(a, vboolf8(embree::True)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator &(const vboolf8& a, const vboolf8& b) { return _mm256_and_ps(a, b); }
  __forceinline vboolf8 operator |(const vboolf8& a, const vboolf8& b) { return _mm256_or_ps (a, b); }
  __forceinline vboolf8 operator ^(const vboolf8& a, const vboolf8& b) { return _mm256_xor_ps(a, b); }

  __forceinline vboolf8 andn(const vboolf8& a, const vboolf8& b) { return _mm256_andnot_ps(b, a); }

  __forceinline vboolf8& operator &=(vboolf8& a, const vboolf8& b) { return a = a & b; }
  __forceinline vboolf8& operator |=(vboolf8& a, const vboolf8& b) { return a = a | b; }
  __forceinline vboolf8& operator ^=(vboolf8& a, const vboolf8& b) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator !=(const vboolf8& a, const vboolf8& b) { return _mm256_xor_ps(a, b); }
  __forceinline vboolf8 operator ==(const vboolf8& a, const vboolf8& b) { return _mm256_xor_ps(_mm256_xor_ps(a,b),vboolf8(embree::True)); }

  __forceinline vboolf8 select(const vboolf8& mask, const vboolf8& t, const vboolf8& f) {
    return _mm256_blendv_ps(f, t, mask); 
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Movement/Shifting/Shuffling Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 unpacklo(const vboolf8& a, const vboolf8& b) { return _mm256_unpacklo_ps(a, b); }
  __forceinline vboolf8 unpackhi(const vboolf8& a, const vboolf8& b) { return _mm256_unpackhi_ps(a, b); }

  template<int i>
  __forceinline vboolf8 shuffle(const vboolf8& v) {
    return _mm256_permute_ps(v, _MM_SHUFFLE(i, i, i, i));
  }

  template<int i0, int i1>
  __forceinline vboolf8 shuffle4(const vboolf8& v) {
    return _mm256_permute2f128_ps(v, v, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1>
  __forceinline vboolf8 shuffle4(const vboolf8& a, const vboolf8& b) {
    return _mm256_permute2f128_ps(a, b, (i1 << 4) | (i0 << 0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vboolf8 shuffle(const vboolf8& v) {
    return _mm256_permute_ps(v, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<int i0, int i1, int i2, int i3>
  __forceinline vboolf8 shuffle(const vboolf8& a, const vboolf8& b) {
    return _mm256_shuffle_ps(a, b, _MM_SHUFFLE(i3, i2, i1, i0));
  }

  template<> __forceinline vboolf8 shuffle<0, 0, 2, 2>(const vboolf8& v) { return _mm256_moveldup_ps(v); }
  template<> __forceinline vboolf8 shuffle<1, 1, 3, 3>(const vboolf8& v) { return _mm256_movehdup_ps(v); }
  template<> __forceinline vboolf8 shuffle<0, 1, 0, 1>(const vboolf8& v) { return _mm256_castpd_ps(_mm256_movedup_pd(_mm256_castps_pd(v))); }

  template<int i> __forceinline vboolf8 insert4(const vboolf8& a, const vboolf4& b) { return _mm256_insertf128_ps(a, b, i); }
  template<int i> __forceinline vboolf4 extract4   (const vboolf8& a) { return _mm256_extractf128_ps(a, i); }
  template<>      __forceinline vboolf4 extract4<0>(const vboolf8& a) { return _mm256_castps256_ps128(a);   }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool reduce_and(const vboolf8& a) { return _mm256_movemask_ps(a) == (unsigned int)0xff; }
  __forceinline bool reduce_or (const vboolf8& a) { return !_mm256_testz_ps(a,a); }

  __forceinline bool all (const vboolf8& a) { return _mm256_movemask_ps(a) == (unsigned int)0xff; }
  __forceinline bool any (const vboolf8& a) { return !_mm256_testz_ps(a,a); }
  __forceinline bool none(const vboolf8& a) { return _mm256_testz_ps(a,a) != 0; }

  __forceinline bool all (const vboolf8& valid, const vboolf8& b) { return all((!valid) | b); }
  __forceinline bool any (const vboolf8& valid, const vboolf8& b) { return any(valid & b); }
  __forceinline bool none(const vboolf8& valid, const vboolf8& b) { return none(valid & b); }

  __forceinline unsigned int movemask(const vboolf8& a) { return _mm256_movemask_ps(a); }
  __forceinline size_t       popcnt  (const vboolf8& a) { return popcnt((size_t)_mm256_movemask_ps(a)); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboolf8& a, size_t index) { return a[index]; }
  __forceinline void set(vboolf8& a, size_t index)       { a[index] = -1; }
  __forceinline void clear(vboolf8& a, size_t index)     { a[index] =  0; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator <<(std::ostream& cout, const vboolf8& a) {
    return cout << "<" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ", "
                       << a[4] << ", " << a[5] << ", " << a[6] << ", " << a[7] << ">";
  }
}
