/*
 * Copyright © 2017  Google, Inc.
 * Copyright © 2019  Facebook, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 * Facebook Author(s): Behdad Esfahbod
 */

#ifndef HB_ALGS_HH
#define HB_ALGS_HH

#include "hb.hh"
#include "hb-meta.hh"
#include "hb-null.hh"
#include "hb-number.hh"

#include <algorithm>
#include <initializer_list>
#include <functional>
#include <new>

/*
 * Flags
 */

/* Enable bitwise ops on enums marked as flags_t */
/* To my surprise, looks like the function resolver is happy to silently cast
 * one enum to another...  So this doesn't provide the type-checking that I
 * originally had in mind... :(.
 *
 * For MSVC warnings, see: https://github.com/harfbuzz/harfbuzz/pull/163
 */
#ifdef _MSC_VER
# pragma warning(disable:4200)
# pragma warning(disable:4800)
#endif
#define HB_MARK_AS_FLAG_T(T) \
	extern "C++" { \
	  static inline constexpr T operator | (T l, T r) { return T ((unsigned) l | (unsigned) r); } \
	  static inline constexpr T operator & (T l, T r) { return T ((unsigned) l & (unsigned) r); } \
	  static inline constexpr T operator ^ (T l, T r) { return T ((unsigned) l ^ (unsigned) r); } \
	  static inline constexpr unsigned operator ~ (T r) { return (~(unsigned) r); } \
	  static inline T& operator |= (T &l, T r) { l = l | r; return l; } \
	  static inline T& operator &= (T& l, T r) { l = l & r; return l; } \
	  static inline T& operator ^= (T& l, T r) { l = l ^ r; return l; } \
	} \
	static_assert (true, "")

/* Useful for set-operations on small enums.
 * For example, for testing "x ∈ {x1, x2, x3}" use:
 * (FLAG_UNSAFE(x) & (FLAG(x1) | FLAG(x2) | FLAG(x3)))
 */
#define FLAG(x) (static_assert_expr ((unsigned)(x) < 32) + (((uint32_t) 1U) << (unsigned)(x)))
#define FLAG_UNSAFE(x) ((unsigned)(x) < 32 ? (((uint32_t) 1U) << (unsigned)(x)) : 0)
#define FLAG_RANGE(x,y) (static_assert_expr ((x) < (y)) + FLAG(y+1) - FLAG(x))
#define FLAG64(x) (static_assert_expr ((unsigned)(x) < 64) + (((uint64_t) 1ULL) << (unsigned)(x)))
#define FLAG64_UNSAFE(x) ((unsigned)(x) < 64 ? (((uint64_t) 1ULL) << (unsigned)(x)) : 0)


/*
 * Big-endian integers.
 */

/* Endian swap, used in Windows related backends */
static inline constexpr uint16_t hb_uint16_swap (uint16_t v)
{ return (v >> 8) | (v << 8); }
static inline constexpr uint32_t hb_uint32_swap (uint32_t v)
{ return (hb_uint16_swap (v) << 16) | hb_uint16_swap (v >> 16); }

#ifndef HB_FAST_INT_ACCESS
#if defined(__OPTIMIZE__) && \
    defined(__BYTE_ORDER) && \
    (__BYTE_ORDER == __BIG_ENDIAN || \
     (__BYTE_ORDER == __LITTLE_ENDIAN && \
      hb_has_builtin(__builtin_bswap16) && \
      hb_has_builtin(__builtin_bswap32)))
#define HB_FAST_INT_ACCESS 1
#else
#define HB_FAST_INT_ACCESS 0
#endif
#endif

template <typename Type, int Bytes = sizeof (Type)>
struct BEInt;
template <typename Type>
struct BEInt<Type, 1>
{
  public:
  BEInt () = default;
  constexpr BEInt (Type V) : v {uint8_t (V)} {}
  constexpr operator Type () const { return v; }
  private: uint8_t v;
};
template <typename Type>
struct BEInt<Type, 2>
{
  struct __attribute__((packed)) packed_uint16_t { uint16_t v; };

  public:
  BEInt () = default;

  BEInt (Type V)
#if HB_FAST_INT_ACCESS
#if __BYTE_ORDER == __LITTLE_ENDIAN
  { ((packed_uint16_t *) v)->v = __builtin_bswap16 (V); }
#else /* __BYTE_ORDER == __BIG_ENDIAN */
  { ((packed_uint16_t *) v)->v = V; }
#endif
#else
    : v {uint8_t ((V >>  8) & 0xFF),
	 uint8_t ((V      ) & 0xFF)} {}
#endif

  constexpr operator Type () const {
#if HB_FAST_INT_ACCESS
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return __builtin_bswap16 (((packed_uint16_t *) v)->v);
#else /* __BYTE_ORDER == __BIG_ENDIAN */
    return ((packed_uint16_t *) v)->v;
#endif
#else
    return (v[0] <<  8)
	 + (v[1]      );
#endif
  }
  private: uint8_t v[2];
};
template <typename Type>
struct BEInt<Type, 3>
{
  static_assert (!std::is_signed<Type>::value, "");
  public:
  BEInt () = default;
  constexpr BEInt (Type V) : v {uint8_t ((V >> 16) & 0xFF),
				uint8_t ((V >>  8) & 0xFF),
				uint8_t ((V      ) & 0xFF)} {}

  constexpr operator Type () const { return (v[0] << 16)
					  + (v[1] <<  8)
					  + (v[2]      ); }
  private: uint8_t v[3];
};
template <typename Type>
struct BEInt<Type, 4>
{
  struct __attribute__((packed)) packed_uint32_t { uint32_t v; };

  public:
  BEInt () = default;

  BEInt (Type V)
#if HB_FAST_INT_ACCESS
#if __BYTE_ORDER == __LITTLE_ENDIAN
  { ((packed_uint32_t *) v)->v = __builtin_bswap32 (V); }
#else /* __BYTE_ORDER == __BIG_ENDIAN */
  { ((packed_uint32_t *) v)->v = V; }
#endif
#else
    : v {uint8_t ((V >> 24) & 0xFF),
	 uint8_t ((V >> 16) & 0xFF),
	 uint8_t ((V >>  8) & 0xFF),
	 uint8_t ((V      ) & 0xFF)} {}
#endif

  constexpr operator Type () const {
#if HB_FAST_INT_ACCESS
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return __builtin_bswap32 (((packed_uint32_t *) v)->v);
#else /* __BYTE_ORDER == __BIG_ENDIAN */
    return ((packed_uint32_t *) v)->v;
#endif
#else
    return (v[0] << 24)
	 + (v[1] << 16)
	 + (v[2] <<  8)
	 + (v[3]      );
#endif
  }
  private: uint8_t v[4];
};

/* Floats. */

/* We want our rounding towards +infinity. */
static inline float
_hb_roundf (float x) { return floorf (x + .5f); }
#define roundf(x) _hb_roundf(x)


/* Encodes three unsigned integers in one 64-bit number.  If the inputs have more than 21 bits,
 * values will be truncated / overlap, and might not decode exactly. */
#define HB_CODEPOINT_ENCODE3(x,y,z) (((uint64_t) (x) << 42) | ((uint64_t) (y) << 21) | (uint64_t) (z))
#define HB_CODEPOINT_DECODE3_1(v) ((hb_codepoint_t) ((v) >> 42))
#define HB_CODEPOINT_DECODE3_2(v) ((hb_codepoint_t) ((v) >> 21) & 0x1FFFFFu)
#define HB_CODEPOINT_DECODE3_3(v) ((hb_codepoint_t) (v) & 0x1FFFFFu)

/* Custom encoding used by hb-ucd. */
#define HB_CODEPOINT_ENCODE3_11_7_14(x,y,z) (((uint32_t) ((x) & 0x07FFu) << 21) | (((uint32_t) (y) & 0x007Fu) << 14) | (uint32_t) ((z) & 0x3FFFu))
#define HB_CODEPOINT_DECODE3_11_7_14_1(v) ((hb_codepoint_t) ((v) >> 21))
#define HB_CODEPOINT_DECODE3_11_7_14_2(v) ((hb_codepoint_t) (((v) >> 14) & 0x007Fu) | 0x0300)
#define HB_CODEPOINT_DECODE3_11_7_14_3(v) ((hb_codepoint_t) (v) & 0x3FFFu)


struct
{
  /* Note.  This is dangerous in that if it's passed an rvalue, it returns rvalue-reference. */
  template <typename T> constexpr auto
  operator () (T&& v) const HB_AUTO_RETURN ( std::forward<T> (v) )
}
HB_FUNCOBJ (hb_identity);
struct
{
  /* Like identity(), but only retains lvalue-references.  Rvalues are returned as rvalues. */
  template <typename T> constexpr T&
  operator () (T& v) const { return v; }

  template <typename T> constexpr hb_remove_reference<T>
  operator () (T&& v) const { return v; }
}
HB_FUNCOBJ (hb_lidentity);
struct
{
  /* Like identity(), but always returns rvalue. */
  template <typename T> constexpr hb_remove_reference<T>
  operator () (T&& v) const { return v; }
}
HB_FUNCOBJ (hb_ridentity);

struct
{
  template <typename T> constexpr bool
  operator () (T&& v) const { return bool (std::forward<T> (v)); }
}
HB_FUNCOBJ (hb_bool);


/* The MIT License

   Copyright (C) 2012 Zilong Tan (eric.zltan@gmail.com)

   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/


// Compression function for Merkle-Damgard construction.
// This function is generated using the framework provided.
#define mix(h) (					\
			(void) ((h) ^= (h) >> 23),		\
			(void) ((h) *= 0x2127599bf4325c37ULL),	\
			(h) ^= (h) >> 47)

static inline uint64_t fasthash64(const void *buf, size_t len, uint64_t seed)
{
	struct __attribute__((packed)) packed_uint64_t { uint64_t v; };
	const uint64_t    m = 0x880355f21e6d1965ULL;
	const packed_uint64_t *pos = (const packed_uint64_t *)buf;
	const packed_uint64_t *end = pos + (len / 8);
	const unsigned char *pos2;
	uint64_t h = seed ^ (len * m);
	uint64_t v;

#ifndef HB_OPTIMIZE_SIZE
	if (((uintptr_t) pos & 7) == 0)
	{
	  while (pos != end)
	  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
	    v  = * (const uint64_t *) (pos++);
#pragma GCC diagnostic pop
	    h ^= mix(v);
	    h *= m;
	  }
	}
	else
#endif
	{
	  while (pos != end)
	  {
	    v  = pos++->v;
	    h ^= mix(v);
	    h *= m;
	  }
	}

	pos2 = (const unsigned char*)pos;
	v = 0;

	switch (len & 7) {
	case 7: v ^= (uint64_t)pos2[6] << 48; HB_FALLTHROUGH;
	case 6: v ^= (uint64_t)pos2[5] << 40; HB_FALLTHROUGH;
	case 5: v ^= (uint64_t)pos2[4] << 32; HB_FALLTHROUGH;
	case 4: v ^= (uint64_t)pos2[3] << 24; HB_FALLTHROUGH;
	case 3: v ^= (uint64_t)pos2[2] << 16; HB_FALLTHROUGH;
	case 2: v ^= (uint64_t)pos2[1] <<  8; HB_FALLTHROUGH;
	case 1: v ^= (uint64_t)pos2[0];
		h ^= mix(v);
		h *= m;
	}

	return mix(h);
}

static inline uint32_t fasthash32(const void *buf, size_t len, uint32_t seed)
{
	// the following trick converts the 64-bit hashcode to Fermat
	// residue, which shall retain information from both the higher
	// and lower parts of hashcode.
        uint64_t h = fasthash64(buf, len, seed);
	return h - (h >> 32);
}

struct
{
  private:

  template <typename T> constexpr auto
  impl (const T& v, hb_priority<2>) const HB_RETURN (uint32_t, hb_deref (v).hash ())

  // Horrible: std:hash() of integers seems to be identity in gcc / clang?!
  // https://github.com/harfbuzz/harfbuzz/pull/4228
  //
  // For performance characteristics see:
  // https://github.com/harfbuzz/harfbuzz/pull/4228#issuecomment-1565079537
  template <typename T,
	    hb_enable_if (std::is_integral<T>::value && sizeof (T) <= sizeof (uint32_t))> constexpr auto
  impl (const T& v, hb_priority<1>) const HB_RETURN (uint32_t, (uint32_t) v * 2654435761u /* Knuh's multiplicative hash */)
  template <typename T,
	    hb_enable_if (std::is_integral<T>::value && sizeof (T) > sizeof (uint32_t))> constexpr auto
  impl (const T& v, hb_priority<1>) const HB_RETURN (uint32_t, (uint32_t) (v ^ (v >> 32)) * 2654435761u /* Knuth's multiplicative hash */)

  template <typename T,
	    hb_enable_if (std::is_floating_point<T>::value)> constexpr auto
  impl (const T& v, hb_priority<1>) const HB_RETURN (uint32_t, fasthash32 (std::addressof (v), sizeof (T), 0xf437ffe6))

  template <typename T> constexpr auto
  impl (const T& v, hb_priority<0>) const HB_RETURN (uint32_t, std::hash<hb_decay<decltype (hb_deref (v))>>{} (hb_deref (v)))

  public:

  template <typename T> constexpr auto
  operator () (const T& v) const HB_RETURN (uint32_t, impl (v, hb_prioritize))
}
HB_FUNCOBJ (hb_hash);


struct
{
  private:

  /* Pointer-to-member-function. */
  template <typename Appl, typename T, typename ...Ts> auto
  impl (Appl&& a, hb_priority<2>, T &&v, Ts&&... ds) const HB_AUTO_RETURN
  ((hb_deref (std::forward<T> (v)).*std::forward<Appl> (a)) (std::forward<Ts> (ds)...))

  /* Pointer-to-member. */
  template <typename Appl, typename T> auto
  impl (Appl&& a, hb_priority<1>, T &&v) const HB_AUTO_RETURN
  ((hb_deref (std::forward<T> (v))).*std::forward<Appl> (a))

  /* Operator(). */
  template <typename Appl, typename ...Ts> auto
  impl (Appl&& a, hb_priority<0>, Ts&&... ds) const HB_AUTO_RETURN
  (hb_deref (std::forward<Appl> (a)) (std::forward<Ts> (ds)...))

  public:

  template <typename Appl, typename ...Ts> auto
  operator () (Appl&& a, Ts&&... ds) const HB_AUTO_RETURN
  (
    impl (std::forward<Appl> (a),
	  hb_prioritize,
	  std::forward<Ts> (ds)...)
  )
}
HB_FUNCOBJ (hb_invoke);

template <unsigned Pos, typename Appl, typename V>
struct hb_partial_t
{
  hb_partial_t (Appl a, V v) : a (a), v (v) {}

  static_assert (Pos > 0, "");

  template <typename ...Ts,
	    unsigned P = Pos,
	    hb_enable_if (P == 1)> auto
  operator () (Ts&& ...ds) -> decltype (hb_invoke (hb_declval (Appl),
						   hb_declval (V),
						   hb_declval (Ts)...))
  {
    return hb_invoke (std::forward<Appl> (a),
		      std::forward<V> (v),
		      std::forward<Ts> (ds)...);
  }
  template <typename T0, typename ...Ts,
	    unsigned P = Pos,
	    hb_enable_if (P == 2)> auto
  operator () (T0&& d0, Ts&& ...ds) -> decltype (hb_invoke (hb_declval (Appl),
							    hb_declval (T0),
							    hb_declval (V),
							    hb_declval (Ts)...))
  {
    return hb_invoke (std::forward<Appl> (a),
		      std::forward<T0> (d0),
		      std::forward<V> (v),
		      std::forward<Ts> (ds)...);
  }

  private:
  hb_reference_wrapper<Appl> a;
  V v;
};
template <unsigned Pos=1, typename Appl, typename V>
auto hb_partial (Appl&& a, V&& v) HB_AUTO_RETURN
(( hb_partial_t<Pos, Appl, V> (a, v) ))

/* The following, HB_PARTIALIZE, macro uses a particular corner-case
 * of C++11 that is not particularly well-supported by all compilers.
 * What's happening is that it's using "this" in a trailing return-type
 * via decltype().  Broken compilers deduce the type of "this" pointer
 * in that context differently from what it resolves to in the body
 * of the function.
 *
 * One probable cause of this is that at the time of trailing return
 * type declaration, "this" points to an incomplete type, whereas in
 * the function body the type is complete.  That doesn't justify the
 * error in any way, but is probably what's happening.
 *
 * In the case of MSVC, we get around this by using C++14 "decltype(auto)"
 * which deduces the type from the actual return statement.  For gcc 4.8
 * we use "+this" instead of "this" which produces an rvalue that seems
 * to be deduced as the same type with this particular compiler, and seem
 * to be fine as default code path as well.
 */
#ifdef _MSC_VER
/* https://github.com/harfbuzz/harfbuzz/issues/1730 */ \
#define HB_PARTIALIZE(Pos) \
  template <typename _T> \
  decltype(auto) operator () (_T&& _v) const \
  { return hb_partial<Pos> (this, std::forward<_T> (_v)); } \
  static_assert (true, "")
#else
/* https://github.com/harfbuzz/harfbuzz/issues/1724 */
#define HB_PARTIALIZE(Pos) \
  template <typename _T> \
  auto operator () (_T&& _v) const HB_AUTO_RETURN \
  (hb_partial<Pos> (+this, std::forward<_T> (_v))) \
  static_assert (true, "")
#endif


struct
{
  private:

  template <typename Pred, typename Val> auto
  impl (Pred&& p, Val &&v, hb_priority<1>) const HB_AUTO_RETURN
  (
    hb_deref (std::forward<Pred> (p)).has (std::forward<Val> (v))
  )

  template <typename Pred, typename Val> auto
  impl (Pred&& p, Val &&v, hb_priority<0>) const HB_AUTO_RETURN
  (
    hb_invoke (std::forward<Pred> (p),
	       std::forward<Val> (v))
  )

  public:

  template <typename Pred, typename Val> auto
  operator () (Pred&& p, Val &&v) const HB_RETURN (bool,
    impl (std::forward<Pred> (p),
	  std::forward<Val> (v),
	  hb_prioritize)
  )
}
HB_FUNCOBJ (hb_has);

struct
{
  private:

  template <typename Pred, typename Val> auto
  impl (Pred&& p, Val &&v, hb_priority<1>) const HB_AUTO_RETURN
  (
    hb_has (std::forward<Pred> (p),
	    std::forward<Val> (v))
  )

  template <typename Pred, typename Val> auto
  impl (Pred&& p, Val &&v, hb_priority<0>) const HB_AUTO_RETURN
  (
    std::forward<Pred> (p) == std::forward<Val> (v)
  )

  public:

  template <typename Pred, typename Val> auto
  operator () (Pred&& p, Val &&v) const HB_RETURN (bool,
    impl (std::forward<Pred> (p),
	  std::forward<Val> (v),
	  hb_prioritize)
  )
}
HB_FUNCOBJ (hb_match);

struct
{
  private:

  template <typename Proj, typename Val> auto
  impl (Proj&& f, Val &&v, hb_priority<2>) const HB_AUTO_RETURN
  (
    hb_deref (std::forward<Proj> (f)).get (std::forward<Val> (v))
  )

  template <typename Proj, typename Val> auto
  impl (Proj&& f, Val &&v, hb_priority<1>) const HB_AUTO_RETURN
  (
    hb_invoke (std::forward<Proj> (f),
	       std::forward<Val> (v))
  )

  template <typename Proj, typename Val> auto
  impl (Proj&& f, Val &&v, hb_priority<0>) const HB_AUTO_RETURN
  (
    std::forward<Proj> (f)[std::forward<Val> (v)]
  )

  public:

  template <typename Proj, typename Val> auto
  operator () (Proj&& f, Val &&v) const HB_AUTO_RETURN
  (
    impl (std::forward<Proj> (f),
	  std::forward<Val> (v),
	  hb_prioritize)
  )
}
HB_FUNCOBJ (hb_get);

struct
{
  private:

  template <typename T1, typename T2> auto
  impl (T1&& v1, T2 &&v2, hb_priority<3>) const HB_AUTO_RETURN
  (
    std::forward<T2> (v2).cmp (std::forward<T1> (v1)) == 0
  )

  template <typename T1, typename T2> auto
  impl (T1&& v1, T2 &&v2, hb_priority<2>) const HB_AUTO_RETURN
  (
    std::forward<T1> (v1).cmp (std::forward<T2> (v2)) == 0
  )

  template <typename T1, typename T2> auto
  impl (T1&& v1, T2 &&v2, hb_priority<1>) const HB_AUTO_RETURN
  (
    std::forward<T1> (v1) == std::forward<T2> (v2)
  )

  template <typename T1, typename T2> auto
  impl (T1&& v1, T2 &&v2, hb_priority<0>) const HB_AUTO_RETURN
  (
    std::forward<T2> (v2) == std::forward<T1> (v1)
  )

  public:

  template <typename T1, typename T2> auto
  operator () (T1&& v1, T2 &&v2) const HB_AUTO_RETURN
  (
    impl (std::forward<T1> (v1),
	  std::forward<T2> (v2),
	  hb_prioritize)
  )
}
HB_FUNCOBJ (hb_equal);

struct
{
  template <typename T> void
  operator () (T& a, T& b) const
  {
    using std::swap; // allow ADL
    swap (a, b);
  }
}
HB_FUNCOBJ (hb_swap);


template <typename T1, typename T2>
struct hb_pair_t
{
  typedef T1 first_t;
  typedef T2 second_t;
  typedef hb_pair_t<T1, T2> pair_t;

  template <typename U1 = T1, typename U2 = T2,
	    hb_enable_if (std::is_default_constructible<U1>::value &&
			  std::is_default_constructible<U2>::value)>
  hb_pair_t () : first (), second () {}
  hb_pair_t (T1 a, T2 b) : first (std::forward<T1> (a)), second (std::forward<T2> (b)) {}

  template <typename Q1, typename Q2,
	    hb_enable_if (hb_is_convertible (T1, Q1) &&
			  hb_is_convertible (T2, Q2))>
  operator hb_pair_t<Q1, Q2> () { return hb_pair_t<Q1, Q2> (first, second); }

  hb_pair_t<T1, T2> reverse () const
  { return hb_pair_t<T1, T2> (second, first); }

  bool operator == (const pair_t& o) const { return first == o.first && second == o.second; }
  bool operator != (const pair_t& o) const { return !(*this == o); }
  bool operator < (const pair_t& o) const { return first < o.first || (first == o.first && second < o.second); }
  bool operator >= (const pair_t& o) const { return !(*this < o); }
  bool operator > (const pair_t& o) const { return first > o.first || (first == o.first && second > o.second); }
  bool operator <= (const pair_t& o) const { return !(*this > o); }

  static int cmp (const void *pa, const void *pb)
  {
    pair_t *a = (pair_t *) pa;
    pair_t *b = (pair_t *) pb;

    if (a->first < b->first) return -1;
    if (a->first > b->first) return +1;
    if (a->second < b->second) return -1;
    if (a->second > b->second) return +1;
    return 0;
  }

  friend void swap (hb_pair_t& a, hb_pair_t& b)
  {
    hb_swap (a.first, b.first);
    hb_swap (a.second, b.second);
  }


  T1 first;
  T2 second;
};
template <typename T1, typename T2> static inline hb_pair_t<T1, T2>
hb_pair (T1&& a, T2&& b) { return hb_pair_t<T1, T2> (a, b); }

typedef hb_pair_t<hb_codepoint_t, hb_codepoint_t> hb_codepoint_pair_t;

struct
{
  template <typename Pair> constexpr typename Pair::first_t
  operator () (const Pair& pair) const { return pair.first; }
}
HB_FUNCOBJ (hb_first);

struct
{
  template <typename Pair> constexpr typename Pair::second_t
  operator () (const Pair& pair) const { return pair.second; }
}
HB_FUNCOBJ (hb_second);

/* Note.  In min/max impl, we can use hb_type_identity<T> for second argument.
 * However, that would silently convert between different-signedness integers.
 * Instead we accept two different types, such that compiler can err if
 * comparing integers of different signedness. */
struct
{
  template <typename T, typename T2> constexpr auto
  operator () (T&& a, T2&& b) const HB_AUTO_RETURN
  (a <= b ? a : b)
}
HB_FUNCOBJ (hb_min);
struct
{
  template <typename T, typename T2> constexpr auto
  operator () (T&& a, T2&& b) const HB_AUTO_RETURN
  (a >= b ? a : b)
}
HB_FUNCOBJ (hb_max);
struct
{
  template <typename T, typename T2, typename T3> constexpr auto
  operator () (T&& x, T2&& min, T3&& max) const HB_AUTO_RETURN
  (hb_min (hb_max (std::forward<T> (x), std::forward<T2> (min)), std::forward<T3> (max)))
}
HB_FUNCOBJ (hb_clamp);

/*
 * Bithacks.
 */

/* Return the number of 1 bits in v. */
template <typename T>
static inline unsigned int
hb_popcount (T v)
{
#if hb_has_builtin(__builtin_popcount)
  if (sizeof (T) <= sizeof (unsigned int))
    return __builtin_popcount (v);
#endif

#if hb_has_builtin(__builtin_popcountl)
  if (sizeof (T) <= sizeof (unsigned long))
    return __builtin_popcountl (v);
#endif

#if hb_has_builtin(__builtin_popcountll)
  if (sizeof (T) <= sizeof (unsigned long long))
    return __builtin_popcountll (v);
#endif

  if (sizeof (T) <= 4)
  {
    /* "HACKMEM 169" */
    uint32_t y;
    y = (v >> 1) &033333333333;
    y = v - y - ((y >>1) & 033333333333);
    return (((y + (y >> 3)) & 030707070707) % 077);
  }

  if (sizeof (T) == 8)
  {
    uint64_t y = (uint64_t) v;
    y -= ((y >> 1) & 0x5555555555555555ull);
    y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);
    return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
  }

  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return hb_popcount<uint64_t> ((uint64_t) v) + hb_popcount ((uint64_t) (v >> shift));
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}

/* Returns the number of bits needed to store number */
template <typename T>
static inline unsigned int
hb_bit_storage (T v)
{
  if (unlikely (!v)) return 0;

#if hb_has_builtin(__builtin_clz)
  if (sizeof (T) <= sizeof (unsigned int))
    return sizeof (unsigned int) * 8 - __builtin_clz (v);
#endif

#if hb_has_builtin(__builtin_clzl)
  if (sizeof (T) <= sizeof (unsigned long))
    return sizeof (unsigned long) * 8 - __builtin_clzl (v);
#endif

#if hb_has_builtin(__builtin_clzll)
  if (sizeof (T) <= sizeof (unsigned long long))
    return sizeof (unsigned long long) * 8 - __builtin_clzll (v);
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || (defined(__MINGW32__) && (__GNUC__ < 4))
  if (sizeof (T) <= sizeof (unsigned int))
  {
    unsigned long where;
    _BitScanReverse (&where, v);
    return 1 + where;
  }
# if defined(_WIN64)
  if (sizeof (T) <= 8)
  {
    unsigned long where;
    _BitScanReverse64 (&where, v);
    return 1 + where;
  }
# endif
#endif

  if (sizeof (T) <= 4)
  {
    /* "bithacks" */
    const unsigned int b[] = {0x2, 0xC, 0xF0, 0xFF00, 0xFFFF0000};
    const unsigned int S[] = {1, 2, 4, 8, 16};
    unsigned int r = 0;
    for (int i = 4; i >= 0; i--)
      if (v & b[i])
      {
	v >>= S[i];
	r |= S[i];
      }
    return r + 1;
  }
  if (sizeof (T) <= 8)
  {
    /* "bithacks" */
    const uint64_t b[] = {0x2ULL, 0xCULL, 0xF0ULL, 0xFF00ULL, 0xFFFF0000ULL, 0xFFFFFFFF00000000ULL};
    const unsigned int S[] = {1, 2, 4, 8, 16, 32};
    unsigned int r = 0;
    for (int i = 5; i >= 0; i--)
      if (v & b[i])
      {
	v >>= S[i];
	r |= S[i];
      }
    return r + 1;
  }
  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return (v >> shift) ? hb_bit_storage<uint64_t> ((uint64_t) (v >> shift)) + shift :
			  hb_bit_storage<uint64_t> ((uint64_t) v);
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}

/* Returns the number of zero bits in the least significant side of v */
template <typename T>
static inline unsigned int
hb_ctz (T v)
{
  if (unlikely (!v)) return 8 * sizeof (T);

#if hb_has_builtin(__builtin_ctz)
  if (sizeof (T) <= sizeof (unsigned int))
    return __builtin_ctz (v);
#endif

#if hb_has_builtin(__builtin_ctzl)
  if (sizeof (T) <= sizeof (unsigned long))
    return __builtin_ctzl (v);
#endif

#if hb_has_builtin(__builtin_ctzll)
  if (sizeof (T) <= sizeof (unsigned long long))
    return __builtin_ctzll (v);
#endif

#if (defined(_MSC_VER) && _MSC_VER >= 1500) || (defined(__MINGW32__) && (__GNUC__ < 4))
  if (sizeof (T) <= sizeof (unsigned int))
  {
    unsigned long where;
    _BitScanForward (&where, v);
    return where;
  }
# if defined(_WIN64)
  if (sizeof (T) <= 8)
  {
    unsigned long where;
    _BitScanForward64 (&where, v);
    return where;
  }
# endif
#endif

  if (sizeof (T) <= 4)
  {
    /* "bithacks" */
    unsigned int c = 32;
    v &= - (int32_t) v;
    if (v) c--;
    if (v & 0x0000FFFF) c -= 16;
    if (v & 0x00FF00FF) c -= 8;
    if (v & 0x0F0F0F0F) c -= 4;
    if (v & 0x33333333) c -= 2;
    if (v & 0x55555555) c -= 1;
    return c;
  }
  if (sizeof (T) <= 8)
  {
    /* "bithacks" */
    unsigned int c = 64;
    v &= - (int64_t) (v);
    if (v) c--;
    if (v & 0x00000000FFFFFFFFULL) c -= 32;
    if (v & 0x0000FFFF0000FFFFULL) c -= 16;
    if (v & 0x00FF00FF00FF00FFULL) c -= 8;
    if (v & 0x0F0F0F0F0F0F0F0FULL) c -= 4;
    if (v & 0x3333333333333333ULL) c -= 2;
    if (v & 0x5555555555555555ULL) c -= 1;
    return c;
  }
  if (sizeof (T) == 16)
  {
    unsigned int shift = 64;
    return (uint64_t) v ? hb_bit_storage<uint64_t> ((uint64_t) v) :
			  hb_bit_storage<uint64_t> ((uint64_t) (v >> shift)) + shift;
  }

  assert (0);
  return 0; /* Shut up stupid compiler. */
}


/*
 * Tiny stuff.
 */

/* ASCII tag/character handling */
static inline bool ISALPHA (unsigned char c)
{ return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
static inline bool ISALNUM (unsigned char c)
{ return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }
static inline bool ISSPACE (unsigned char c)
{ return c == ' ' || c =='\f'|| c =='\n'|| c =='\r'|| c =='\t'|| c =='\v'; }
static inline unsigned char TOUPPER (unsigned char c)
{ return (c >= 'a' && c <= 'z') ? c - 'a' + 'A' : c; }
static inline unsigned char TOLOWER (unsigned char c)
{ return (c >= 'A' && c <= 'Z') ? c - 'A' + 'a' : c; }
static inline bool ISHEX (unsigned char c)
{ return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
static inline unsigned char TOHEX (uint8_t c)
{ return (c & 0xF) <= 9 ? (c & 0xF) + '0' : (c & 0xF) + 'a' - 10; }
static inline uint8_t FROMHEX (unsigned char c)
{ return (c >= '0' && c <= '9') ? c - '0' : TOLOWER (c) - 'a' + 10; }

static inline unsigned int DIV_CEIL (const unsigned int a, unsigned int b)
{ return (a + (b - 1)) / b; }


#undef  ARRAY_LENGTH
template <typename Type, unsigned int n>
static inline unsigned int ARRAY_LENGTH (const Type (&)[n]) { return n; }
/* A const version, but does not detect erratically being called on pointers. */
#define ARRAY_LENGTH_CONST(__array) ((signed int) (sizeof (__array) / sizeof (__array[0])))


static inline void *
hb_memcpy (void *__restrict dst, const void *__restrict src, size_t len)
{
  /* It's illegal to pass 0 as size to memcpy. */
  if (unlikely (!len)) return dst;
  return memcpy (dst, src, len);
}

static inline int
hb_memcmp (const void *a, const void *b, unsigned int len)
{
  /* It's illegal to pass NULL to memcmp(), even if len is zero.
   * So, wrap it.
   * https://sourceware.org/bugzilla/show_bug.cgi?id=23878 */
  if (unlikely (!len)) return 0;
  return memcmp (a, b, len);
}

static inline void *
hb_memset (void *s, int c, unsigned int n)
{
  /* It's illegal to pass NULL to memset(), even if n is zero. */
  if (unlikely (!n)) return s;
  return memset (s, c, n);
}

static inline unsigned int
hb_ceil_to_4 (unsigned int v)
{
  return ((v - 1) | 3) + 1;
}

template <typename T> static inline bool
hb_in_range (T u, T lo, T hi)
{
  static_assert (!std::is_signed<T>::value, "");

  /* The casts below are important as if T is smaller than int,
   * the subtract results will become a signed int! */
  return (T)(u - lo) <= (T)(hi - lo);
}
template <typename T> static inline bool
hb_in_ranges (T u, T lo1, T hi1)
{
  return hb_in_range (u, lo1, hi1);
}
template <typename T, typename ...Ts> static inline bool
hb_in_ranges (T u, T lo1, T hi1, Ts... ds)
{
  return hb_in_range<T> (u, lo1, hi1) || hb_in_ranges<T> (u, ds...);
}


/*
 * Overflow checking.
 */

static inline bool
hb_unsigned_mul_overflows (unsigned int count, unsigned int size, unsigned *result = nullptr)
{
#if hb_has_builtin(__builtin_mul_overflow)
  unsigned stack_result;
  if (!result)
    result = &stack_result;
  return __builtin_mul_overflow (count, size, result);
#endif

  if (result)
    *result = count * size;
  return (size > 0) && (count >= ((unsigned int) -1) / size);
}


/*
 * Sort and search.
 */

template <typename K, typename V, typename ...Ts>
static int
_hb_cmp_method (const void *pkey, const void *pval, Ts... ds)
{
  const K& key = * (const K*) pkey;
  const V& val = * (const V*) pval;

  return val.cmp (key, ds...);
}

template <typename V, typename K, typename ...Ts>
static inline bool
hb_bsearch_impl (unsigned *pos, /* Out */
		 const K& key,
		 V* base, size_t nmemb, size_t stride,
		 int (*compar)(const void *_key, const void *_item, Ts... _ds),
		 Ts... ds)
{
  /* This is our *only* bsearch implementation. */

  int min = 0, max = (int) nmemb - 1;
  while (min <= max)
  {
    int mid = ((unsigned int) min + (unsigned int) max) / 2;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
    V* p = (V*) (((const char *) base) + (mid * stride));
#pragma GCC diagnostic pop
    int c = compar ((const void *) std::addressof (key), (const void *) p, ds...);
    if (c < 0)
      max = mid - 1;
    else if (c > 0)
      min = mid + 1;
    else
    {
      *pos = mid;
      return true;
    }
  }
  *pos = min;
  return false;
}

template <typename V, typename K>
static inline V*
hb_bsearch (const K& key, V* base,
	    size_t nmemb, size_t stride = sizeof (V),
	    int (*compar)(const void *_key, const void *_item) = _hb_cmp_method<K, V>)
{
  unsigned pos;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
  return hb_bsearch_impl (&pos, key, base, nmemb, stride, compar) ?
	 (V*) (((const char *) base) + (pos * stride)) : nullptr;
#pragma GCC diagnostic pop
}
template <typename V, typename K, typename ...Ts>
static inline V*
hb_bsearch (const K& key, V* base,
	    size_t nmemb, size_t stride,
	    int (*compar)(const void *_key, const void *_item, Ts... _ds),
	    Ts... ds)
{
  unsigned pos;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
  return hb_bsearch_impl (&pos, key, base, nmemb, stride, compar, ds...) ?
	 (V*) (((const char *) base) + (pos * stride)) : nullptr;
#pragma GCC diagnostic pop
}


/* From https://github.com/noporpoise/sort_r
   Feb 5, 2019 (c8c65c1e)
   Modified to support optional argument using templates */

/* Isaac Turner 29 April 2014 Public Domain */

/*
hb_qsort function to be exported.
Parameters:
  base is the array to be sorted
  nel is the number of elements in the array
  width is the size in bytes of each element of the array
  compar is the comparison function
  arg (optional) is a pointer to be passed to the comparison function

void hb_qsort(void *base, size_t nel, size_t width,
              int (*compar)(const void *_a, const void *_b, [void *_arg]),
              [void *arg]);
*/

#define SORT_R_SWAP(a,b,tmp) ((void) ((tmp) = (a)), (void) ((a) = (b)), (b) = (tmp))

/* swap a and b */
/* a and b must not be equal! */
static inline void sort_r_swap(char *__restrict a, char *__restrict b,
                               size_t w)
{
  char tmp, *end = a+w;
  for(; a < end; a++, b++) { SORT_R_SWAP(*a, *b, tmp); }
}

/* swap a, b iff a>b */
/* a and b must not be equal! */
/* __restrict is same as restrict but better support on old machines */
template <typename ...Ts>
static inline int sort_r_cmpswap(char *__restrict a,
                                 char *__restrict b, size_t w,
                                 int (*compar)(const void *_a,
                                               const void *_b,
                                               Ts... _ds),
                                 Ts... ds)
{
  if(compar(a, b, ds...) > 0) {
    sort_r_swap(a, b, w);
    return 1;
  }
  return 0;
}

/*
Swap consecutive blocks of bytes of size na and nb starting at memory addr ptr,
with the smallest swap so that the blocks are in the opposite order. Blocks may
be internally re-ordered e.g.
  12345ab  ->   ab34512
  123abc   ->   abc123
  12abcde  ->   deabc12
*/
static inline void sort_r_swap_blocks(char *ptr, size_t na, size_t nb)
{
  if(na > 0 && nb > 0) {
    if(na > nb) { sort_r_swap(ptr, ptr+na, nb); }
    else { sort_r_swap(ptr, ptr+nb, na); }
  }
}

/* Implement recursive quicksort ourselves */
/* Note: quicksort is not stable, equivalent values may be swapped */
template <typename ...Ts>
static inline void sort_r_simple(void *base, size_t nel, size_t w,
                                 int (*compar)(const void *_a,
                                               const void *_b,
                                               Ts... _ds),
                                 Ts... ds)
{
  char *b = (char *)base, *end = b + nel*w;

  /* for(size_t i=0; i<nel; i++) {printf("%4i", *(int*)(b + i*sizeof(int)));}
  printf("\n"); */

  if(nel < 10) {
    /* Insertion sort for arbitrarily small inputs */
    char *pi, *pj;
    for(pi = b+w; pi < end; pi += w) {
      for(pj = pi; pj > b && sort_r_cmpswap(pj-w,pj,w,compar,ds...); pj -= w) {}
    }
  }
  else
  {
    /* nel > 9; Quicksort */

    int cmp;
    char *pl, *ple, *pr, *pre, *pivot;
    char *last = b+w*(nel-1), *tmp;

    /*
    Use median of second, middle and second-last items as pivot.
    First and last may have been swapped with pivot and therefore be extreme
    */
    char *l[3];
    l[0] = b + w;
    l[1] = b+w*(nel/2);
    l[2] = last - w;

    /* printf("pivots: %i, %i, %i\n", *(int*)l[0], *(int*)l[1], *(int*)l[2]); */

    if(compar(l[0],l[1],ds...) > 0) { SORT_R_SWAP(l[0], l[1], tmp); }
    if(compar(l[1],l[2],ds...) > 0) {
      SORT_R_SWAP(l[1], l[2], tmp);
      if(compar(l[0],l[1],ds...) > 0) { SORT_R_SWAP(l[0], l[1], tmp); }
    }

    /* swap mid value (l[1]), and last element to put pivot as last element */
    if(l[1] != last) { sort_r_swap(l[1], last, w); }

    /*
    pl is the next item on the left to be compared to the pivot
    pr is the last item on the right that was compared to the pivot
    ple is the left position to put the next item that equals the pivot
    ple is the last right position where we put an item that equals the pivot
                                           v- end (beyond the array)
      EEEEEELLLLLLLLuuuuuuuuGGGGGGGEEEEEEEE.
      ^- b  ^- ple  ^- pl   ^- pr  ^- pre ^- last (where the pivot is)
    Pivot comparison key:
      E = equal, L = less than, u = unknown, G = greater than, E = equal
    */
    pivot = last;
    ple = pl = b;
    pre = pr = last;

    /*
    Strategy:
    Loop into the list from the left and right at the same time to find:
    - an item on the left that is greater than the pivot
    - an item on the right that is less than the pivot
    Once found, they are swapped and the loop continues.
    Meanwhile items that are equal to the pivot are moved to the edges of the
    array.
    */
    while(pl < pr) {
      /* Move left hand items which are equal to the pivot to the far left.
         break when we find an item that is greater than the pivot */
      for(; pl < pr; pl += w) {
        cmp = compar(pl, pivot, ds...);
        if(cmp > 0) { break; }
        else if(cmp == 0) {
          if(ple < pl) { sort_r_swap(ple, pl, w); }
          ple += w;
        }
      }
      /* break if last batch of left hand items were equal to pivot */
      if(pl >= pr) { break; }
      /* Move right hand items which are equal to the pivot to the far right.
         break when we find an item that is less than the pivot */
      for(; pl < pr; ) {
        pr -= w; /* Move right pointer onto an unprocessed item */
        cmp = compar(pr, pivot, ds...);
        if(cmp == 0) {
          pre -= w;
          if(pr < pre) { sort_r_swap(pr, pre, w); }
        }
        else if(cmp < 0) {
          if(pl < pr) { sort_r_swap(pl, pr, w); }
          pl += w;
          break;
        }
      }
    }

    pl = pr; /* pr may have gone below pl */

    /*
    Now we need to go from: EEELLLGGGGEEEE
                        to: LLLEEEEEEEGGGG
    Pivot comparison key:
      E = equal, L = less than, u = unknown, G = greater than, E = equal
    */
    sort_r_swap_blocks(b, ple-b, pl-ple);
    sort_r_swap_blocks(pr, pre-pr, end-pre);

    /*for(size_t i=0; i<nel; i++) {printf("%4i", *(int*)(b + i*sizeof(int)));}
    printf("\n");*/

    sort_r_simple(b, (pl-ple)/w, w, compar, ds...);
    sort_r_simple(end-(pre-pr), (pre-pr)/w, w, compar, ds...);
  }
}

static inline void
hb_qsort (void *base, size_t nel, size_t width,
	  int (*compar)(const void *_a, const void *_b))
{
#if defined(__OPTIMIZE_SIZE__) && !defined(HB_USE_INTERNAL_QSORT)
  qsort (base, nel, width, compar);
#else
  sort_r_simple (base, nel, width, compar);
#endif
}

static inline void
hb_qsort (void *base, size_t nel, size_t width,
	  int (*compar)(const void *_a, const void *_b, void *_arg),
	  void *arg)
{
#ifdef HAVE_GNU_QSORT_R
  qsort_r (base, nel, width, compar, arg);
#else
  sort_r_simple (base, nel, width, compar, arg);
#endif
}


template <typename T, typename T2, typename T3 = int> static inline void
hb_stable_sort (T *array, unsigned int len, int(*compar)(const T2 *, const T2 *), T3 *array2 = nullptr)
{
  static_assert (hb_is_trivially_copy_assignable (T), "");
  static_assert (hb_is_trivially_copy_assignable (T3), "");

  for (unsigned int i = 1; i < len; i++)
  {
    unsigned int j = i;
    while (j && compar (&array[j - 1], &array[i]) > 0)
      j--;
    if (i == j)
      continue;
    /* Move item i to occupy place for item j, shift what's in between. */
    {
      T t = array[i];
      memmove (&array[j + 1], &array[j], (i - j) * sizeof (T));
      array[j] = t;
    }
    if (array2)
    {
      T3 t = array2[i];
      memmove (&array2[j + 1], &array2[j], (i - j) * sizeof (T3));
      array2[j] = t;
    }
  }
}

static inline hb_bool_t
hb_codepoint_parse (const char *s, unsigned int len, int base, hb_codepoint_t *out)
{
  unsigned int v;
  const char *p = s;
  const char *end = p + len;
  if (unlikely (!hb_parse_uint (&p, end, &v, true/* whole buffer */, base)))
    return false;

  *out = v;
  return true;
}


/* Operators. */

struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (a & b)
}
HB_FUNCOBJ (hb_bitwise_and);
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (a | b)
}
HB_FUNCOBJ (hb_bitwise_or);
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (a ^ b)
}
HB_FUNCOBJ (hb_bitwise_xor);
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (~a & b)
}
HB_FUNCOBJ (hb_bitwise_lt);
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (a & ~b)
}
HB_FUNCOBJ (hb_bitwise_gt); // aka sub
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (~a | b)
}
HB_FUNCOBJ (hb_bitwise_le);
struct
{ HB_PARTIALIZE(2);
  template <typename T> constexpr auto
  operator () (const T &a, const T &b) const HB_AUTO_RETURN (a | ~b)
}
HB_FUNCOBJ (hb_bitwise_ge);
struct
{
  template <typename T> constexpr auto
  operator () (const T &a) const HB_AUTO_RETURN (~a)
}
HB_FUNCOBJ (hb_bitwise_neg);

struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (a + b)
}
HB_FUNCOBJ (hb_add);
struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (a - b)
}
HB_FUNCOBJ (hb_sub);
struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (b - a)
}
HB_FUNCOBJ (hb_rsub);
struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (a * b)
}
HB_FUNCOBJ (hb_mul);
struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (a / b)
}
HB_FUNCOBJ (hb_div);
struct
{ HB_PARTIALIZE(2);
  template <typename T, typename T2> constexpr auto
  operator () (const T &a, const T2 &b) const HB_AUTO_RETURN (a % b)
}
HB_FUNCOBJ (hb_mod);
struct
{
  template <typename T> constexpr auto
  operator () (const T &a) const HB_AUTO_RETURN (+a)
}
HB_FUNCOBJ (hb_pos);
struct
{
  template <typename T> constexpr auto
  operator () (const T &a) const HB_AUTO_RETURN (-a)
}
HB_FUNCOBJ (hb_neg);
struct
{
  template <typename T> constexpr auto
  operator () (T &a) const HB_AUTO_RETURN (++a)
}
HB_FUNCOBJ (hb_inc);
struct
{
  template <typename T> constexpr auto
  operator () (T &a) const HB_AUTO_RETURN (--a)
}
HB_FUNCOBJ (hb_dec);


/* Adapted from kurbo implementation with extra parameters added,
 * and finding for a particular range instead of 0.
 *
 * For documentation and implementation see:
 *
 * [ITP method]: https://en.wikipedia.org/wiki/ITP_Method
 * [An Enhancement of the Bisection Method Average Performance Preserving Minmax Optimality]: https://dl.acm.org/doi/10.1145/3423597
 * https://docs.rs/kurbo/0.8.1/kurbo/common/fn.solve_itp.html
 * https://github.com/linebender/kurbo/blob/fd839c25ea0c98576c7ce5789305822675a89938/src/common.rs#L162-L248
 */
template <typename func_t>
double solve_itp (func_t f,
		  double a, double b,
		  double epsilon,
		  double min_y, double max_y,
		  double &ya, double &yb, double &y)
{
  unsigned n1_2 = (unsigned) (hb_max (ceil (log2 ((b - a) / epsilon)) - 1.0, 0.0));
  const unsigned n0 = 1; // Hardwired
  const double k1 = 0.2 / (b - a); // Hardwired.
  unsigned nmax = n0 + n1_2;
  double scaled_epsilon = epsilon * double (1llu << nmax);
  double _2_epsilon = 2.0 * epsilon;
  while (b - a > _2_epsilon)
  {
    double x1_2 = 0.5 * (a + b);
    double r = scaled_epsilon - 0.5 * (b - a);
    double xf = (yb * a - ya * b) / (yb - ya);
    double sigma = x1_2 - xf;
    double b_a = b - a;
    // This has k2 = 2 hardwired for efficiency.
    double b_a_k2 = b_a * b_a;
    double delta = k1 * b_a_k2;
    int sigma_sign = sigma >= 0 ? +1 : -1;
    double xt = delta <= fabs (x1_2 - xf) ? xf + delta * sigma_sign : x1_2;
    double xitp = fabs (xt - x1_2) <= r ? xt : x1_2 - r * sigma_sign;
    double yitp = f (xitp);
    if (yitp > max_y)
    {
      b = xitp;
      yb = yitp;
    }
    else if (yitp < min_y)
    {
      a = xitp;
      ya = yitp;
    }
    else
    {
      y = yitp;
      return xitp;
    }
    scaled_epsilon *= 0.5;
  }
  return 0.5 * (a + b);
}


#endif /* HB_ALGS_HH */
