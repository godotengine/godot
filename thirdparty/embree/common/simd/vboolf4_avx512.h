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
  /* 4-wide AVX-512 bool type */
  template<>
  struct vboolf<4>
  {
    typedef vboolf4 Bool;
    typedef vint4   Int;

    enum { size = 4 }; // number of SIMD elements
    __mmask8 v;        // data

    ////////////////////////////////////////////////////////////////////////////////
    /// Constructors, Assignment & Cast Operators
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf() {}
    __forceinline vboolf(const vboolf4& t) { v = t.v; }
    __forceinline vboolf4& operator =(const vboolf4& f) { v = f.v; return *this; }

    __forceinline vboolf(const __mmask8 &t) { v = t; }
    __forceinline operator __mmask8() const { return v; }

    __forceinline vboolf(bool b) { v = b ? 0xf : 0x0; }
    __forceinline vboolf(int t)  { v = (__mmask8)t; }
    __forceinline vboolf(unsigned int t) { v = (__mmask8)t; }

    __forceinline vboolf(bool a, bool b, bool c, bool d)
      : v((__mmask8)((int(d) << 3) | (int(c) << 2) | (int(b) << 1) | int(a))) {}

    /* return int8 mask */
    __forceinline __m128i mask8() const {
      return _mm_movm_epi8(v);
    }

    /* return int32 mask */
    __forceinline __m128i mask32() const {
      return _mm_movm_epi32(v);
    }

    /* return int64 mask */
    __forceinline __m256i mask64() const {
      return _mm256_movm_epi64(v);
    }

    ////////////////////////////////////////////////////////////////////////////////
    /// Constants
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline vboolf(FalseTy) : v(0x0) {}
    __forceinline vboolf(TrueTy)  : v(0xf) {}

    ////////////////////////////////////////////////////////////////////////////////
    /// Array Access
    ////////////////////////////////////////////////////////////////////////////////

    __forceinline bool operator [](size_t index) const {
      assert(index < 4); return (mm512_mask2int(v) >> index) & 1;
    }
  };

  ////////////////////////////////////////////////////////////////////////////////
  /// Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf4 operator !(const vboolf4& a) { return _mm512_kandn(a, 0xf); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf4 operator &(const vboolf4& a, const vboolf4& b) { return _mm512_kand(a, b); }
  __forceinline vboolf4 operator |(const vboolf4& a, const vboolf4& b) { return _mm512_kor(a, b); }
  __forceinline vboolf4 operator ^(const vboolf4& a, const vboolf4& b) { return _mm512_kxor(a, b); }

  __forceinline vboolf4 andn(const vboolf4& a, const vboolf4& b) { return _mm512_kandn(b, a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Assignment Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf4& operator &=(vboolf4& a, const vboolf4& b) { return a = a & b; }
  __forceinline vboolf4& operator |=(vboolf4& a, const vboolf4& b) { return a = a | b; }
  __forceinline vboolf4& operator ^=(vboolf4& a, const vboolf4& b) { return a = a ^ b; }

  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators + Select
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline vboolf4 operator !=(const vboolf4& a, const vboolf4& b) { return _mm512_kxor(a, b); }
  __forceinline vboolf4 operator ==(const vboolf4& a, const vboolf4& b) { return _mm512_kand(_mm512_kxnor(a, b), 0xf); }

  __forceinline vboolf4 select(const vboolf4& s, const vboolf4& a, const vboolf4& b) {
    return _mm512_kor(_mm512_kand(s, a), _mm512_kandn(s, b));
  }

  ////////////////////////////////////////////////////////////////////////////////
  /// Reduction Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline int all (const vboolf4& a) { return a.v == 0xf; }
  __forceinline int any (const vboolf4& a) { return _mm512_kortestz(a, a) == 0; }
  __forceinline int none(const vboolf4& a) { return _mm512_kortestz(a, a) != 0; }

  __forceinline int all (const vboolf4& valid, const vboolf4& b) { return all((!valid) | b); }
  __forceinline int any (const vboolf4& valid, const vboolf4& b) { return any(valid & b); }
  __forceinline int none(const vboolf4& valid, const vboolf4& b) { return none(valid & b); }

  __forceinline size_t movemask(const vboolf4& a) { return _mm512_kmov(a); }
  __forceinline size_t popcnt  (const vboolf4& a) { return popcnt(a.v); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Conversion Operations
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline unsigned int toInt(const vboolf4& a) { return mm512_mask2int(a); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Get/Set Functions
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline bool get(const vboolf4& a, size_t index) { assert(index < 4); return (toInt(a) >> index) & 1; }
  __forceinline void set(vboolf4& a, size_t index)       { assert(index < 4); a |= 1 << index; }
  __forceinline void clear(vboolf4& a, size_t index)     { assert(index < 4); a = andn(a, 1 << index); }

  ////////////////////////////////////////////////////////////////////////////////
  /// Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  __forceinline embree_ostream operator <<(embree_ostream cout, const vboolf4& a)
  {
    cout << "<";
    for (size_t i=0; i<4; i++) {
      if ((a.v >> i) & 1) cout << "1"; else cout << "0";
    }
    return cout << ">";
  }
}

#undef vboolf
#undef vboold
#undef vint
#undef vuint
#undef vllong
#undef vfloat
#undef vdouble
