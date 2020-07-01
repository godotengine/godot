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
  /* 16-wide AVX-512 bool type */
  template<>
  struct vboolf<16>
  {
    typedef vboolf16 Bool;
    typedef vint16   Int;
    typedef vfloat16 Float;

    enum { size = 16 }; // number of SIMD elements
    __mmask16 v;        // data
    
    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////
    
    __forceinline vboolf() {}
    __forceinline vboolf(const vboolf16& t) { v = t.v; }
    __forceinline vboolf16& operator =(const vboolf16& f) { v = f.v; return *this; }

    __forceinline vboolf(const __mmask16& t) { v = t; }
    __forceinline operator __mmask16() const { return v; }
    
    __forceinline vboolf(bool b) { v = b ? 0xFFFF : 0x0000; }
    __forceinline vboolf(int t) { v = (__mmask16)t; }
    __forceinline vboolf(unsigned int t) { v = (__mmask16)t; }

    /* return int8 mask */
    __forceinline __m128i mask8() const {
#if defined(__AVX512BW__)
      return _mm_movm_epi8(v);
#else
      const __m512i f = _mm512_set1_epi32(0);
      const __m512i t = _mm512_set1_epi32(-1);
      const __m512i m =  _mm512_mask_or_epi32(f,v,t,t);
      return _mm512_cvtepi32_epi8(m);
#endif
    }

    /* return int32 mask */
    __forceinline __m512i mask32() const {
#if defined(__AVX512DQ__)
      return _mm512_movm_epi32(v);
#else
      const __m512i f = _mm512_set1_epi32(0);
      const __m512i t = _mm512_set1_epi32(-1);
      return _mm512_mask_or_epi32(f,v,t,t);
#endif
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf(FalseTy) : v(0x0000) {}
    __forceinline vboolf(TrueTy)  : v(0xffff) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////
  
    __forceinline bool operator [](size_t index) const {
      assert(index < 16); return (mm512_mask2int(v) >> index) & 1;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vboolf16 operator !(const vboolf16& a) { return _mm512_knot(a); }
  
   ////////////////////////////////////////////////////////////////////////////////
   /// Binary Operators
   ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vboolf16 operator &(const vboolf16& a, const vboolf16& b) { return _mm512_kand(a,b); }
  __forceinline vboolf16 operator |(const vboolf16& a, const vboolf16& b) { return _mm512_kor(a,b); }
  __forceinline vboolf16 operator ^(const vboolf16& a, const vboolf16& b) { return _mm512_kxor(a,b); }

  __forceinline vboolf16 andn(const vboolf16& a, const vboolf16& b) { return _mm512_kandn(b,a); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vboolf16& operator &=(vboolf16& a, const vboolf16& b) { return a = a & b; }
  __forceinline vboolf16& operator |=(vboolf16& a, const vboolf16& b) { return a = a | b; }
  __forceinline vboolf16& operator ^=(vboolf16& a, const vboolf16& b) { return a = a ^ b; }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline vboolf16 operator !=(const vboolf16& a, const vboolf16& b) { return _mm512_kxor(a, b); }
  __forceinline vboolf16 operator ==(const vboolf16& a, const vboolf16& b) { return _mm512_kxnor(a, b); }
  
  __forceinline vboolf16 select(const vboolf16& s, const vboolf16& a, const vboolf16& b) {
    return _mm512_kor(_mm512_kand(s,a),_mm512_kandn(s,b));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////
  
  __forceinline int all (const vboolf16& a) { return  _mm512_kortestc(a,a) != 0; }
  __forceinline int any (const vboolf16& a) { return  _mm512_kortestz(a,a) == 0; }
  __forceinline int none(const vboolf16& a) { return  _mm512_kortestz(a,a) != 0; }

  __forceinline int all (const vboolf16& valid, const vboolf16& b) { return all((!valid) | b); }
  __forceinline int any (const vboolf16& valid, const vboolf16& b) { return any(valid & b); }
  __forceinline int none(const vboolf16& valid, const vboolf16& b) { return none(valid & b); }
  
  __forceinline size_t movemask(const vboolf16& a) { return _mm512_kmov(a); }
  __forceinline size_t popcnt  (const vboolf16& a) { return popcnt(a.v); }
  
  ////////////////////////////////////////////////////////////////////////////////
  /// Convertion Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline unsigned int toInt (const vboolf16& a) { return mm512_mask2int(a); }
  __forceinline vboolf16     toMask(const int& a)      { return mm512_int2mask(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboolf16& a, size_t index) { assert(index < 16); return (toInt(a) >> index) & 1; }
  __forceinline void set(vboolf16& a, size_t index)       { assert(index < 16); a |= 1 << index; }
  __forceinline void clear(vboolf16& a, size_t index)     { assert(index < 16); a = andn(a, 1 << index); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////
  
  inline std::ostream& operator <<(std::ostream& cout, const vboolf16& a)
  {
    cout << "<";
    for (size_t i=0; i<16; i++) {
      if ((a.v >> i) & 1) cout << "1"; else cout << "0";
    }
    return cout << ">";
  }
}
