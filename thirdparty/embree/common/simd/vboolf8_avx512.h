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
  /* 8-wide AVX-512 bool type */
  template<>
  struct vboolf<8>
  {
    typedef vboolf8 Bool;
    typedef vint8   Int;

    enum { size = 8 }; // number of SIMD elements
    __mmask8 v;        // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf() {}
    __forceinline vboolf(const vboolf8& t) { v = t.v; }
    __forceinline vboolf8& operator =(const vboolf8& f) { v = f.v; return *this; }

    __forceinline vboolf(const __mmask8 &t) { v = t; }
    __forceinline operator __mmask8() const { return v; }

    __forceinline vboolf(bool b) { v = b ? 0xff : 0x00; }
    __forceinline vboolf(int t)  { v = (__mmask8)t; }
    __forceinline vboolf(unsigned int t) { v = (__mmask8)t; }

    __forceinline vboolf(bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h)
      : v((__mmask8)((int(h) << 7) | (int(g) << 6) | (int(f) << 5) | (int(e) << 4) | (int(d) << 3) | (int(c) << 2) | (int(b) << 1) | int(a))) {}

    /* return int8 mask */
    __forceinline __m128i mask8() const {
      return _mm_movm_epi8(v);
    }

    /* return int32 mask */
    __forceinline __m256i mask32() const {
      return _mm256_movm_epi32(v);
    }

    /* return int64 mask */
    __forceinline __m512i mask64() const {
      return _mm512_movm_epi64(v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf(FalseTy) : v(0x00) {}
    __forceinline vboolf(TrueTy)  : v(0xff) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool operator [](size_t index) const {
      assert(index < 8); return (mm512_mask2int(v) >> index) & 1;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator !(const vboolf8& a) { return _mm512_knot(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator &(const vboolf8& a, const vboolf8& b) { return _mm512_kand(a, b); }
  __forceinline vboolf8 operator |(const vboolf8& a, const vboolf8& b) { return _mm512_kor(a, b); }
  __forceinline vboolf8 operator ^(const vboolf8& a, const vboolf8& b) { return _mm512_kxor(a, b); }

  __forceinline vboolf8 andn(const vboolf8& a, const vboolf8& b) { return _mm512_kandn(b, a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8& operator &=(vboolf8& a, const vboolf8& b) { return a = a & b; }
  __forceinline vboolf8& operator |=(vboolf8& a, const vboolf8& b) { return a = a | b; }
  __forceinline vboolf8& operator ^=(vboolf8& a, const vboolf8& b) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf8 operator !=(const vboolf8& a, const vboolf8& b) { return _mm512_kxor(a, b); }
  __forceinline vboolf8 operator ==(const vboolf8& a, const vboolf8& b) { return _mm512_kxnor(a, b); }

  __forceinline vboolf8 select(const vboolf8& s, const vboolf8& a, const vboolf8& b) {
    return _mm512_kor(_mm512_kand(s, a), _mm512_kandn(s, b));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline int all (const vboolf8& a) { return a.v == 0xff; }
  __forceinline int any (const vboolf8& a) { return _mm512_kortestz(a, a) == 0; }
  __forceinline int none(const vboolf8& a) { return _mm512_kortestz(a, a) != 0; }

  __forceinline int all (const vboolf8& valid, const vboolf8& b) { return all((!valid) | b); }
  __forceinline int any (const vboolf8& valid, const vboolf8& b) { return any(valid & b); }
  __forceinline int none(const vboolf8& valid, const vboolf8& b) { return none(valid & b); }

  __forceinline size_t movemask(const vboolf8& a) { return _mm512_kmov(a); }
  __forceinline size_t popcnt  (const vboolf8& a) { return popcnt(a.v); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Conversion Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline unsigned int toInt(const vboolf8& a) { return mm512_mask2int(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboolf8& a, size_t index) { assert(index < 8); return (toInt(a) >> index) & 1; }
  __forceinline void set(vboolf8& a, size_t index)       { assert(index < 8); a |= 1 << index; }
  __forceinline void clear(vboolf8& a, size_t index)     { assert(index < 8); a = andn(a, 1 << index); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  inline std::ostream& operator <<(std::ostream& cout, const vboolf8& a)
  {
    cout << "<";
    for (size_t i=0; i<8; i++) {
      if ((a.v >> i) & 1) cout << "1"; else cout << "0";
    }
    return cout << ">";
  }
}
