/*
 * Copyright © 2007,2008,2009,2010  Red Hat, Inc.
 * Copyright © 2012  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OPEN_TYPE_HH
#define HB_OPEN_TYPE_HH

#include "hb.hh"
#include "hb-blob.hh"
#include "hb-face.hh"
#include "hb-machinery.hh"
#include "hb-meta.hh"
#include "hb-subset.hh"


namespace OT {


/*
 *
 * The OpenType Font File: Data Types
 */


/* "The following data types are used in the OpenType font file.
 *  All OpenType fonts use Motorola-style byte ordering (Big Endian):" */

/*
 * Int types
 */

/* Integer types in big-endian order and no alignment requirement */
template <typename Type,
	  unsigned int Size = sizeof (Type)>
struct IntType
{
  typedef Type type;

  IntType () = default;
  explicit constexpr IntType (Type V) : v {V} {}
  IntType& operator = (Type i) { v = i; return *this; }
  /* For reason we define cast out operator for signed/unsigned, instead of Type, see:
   * https://github.com/harfbuzz/harfbuzz/pull/2875/commits/09836013995cab2b9f07577a179ad7b024130467 */
  operator typename std::conditional<std::is_signed<Type>::value, signed, unsigned>::type () const { return v; }

  bool operator == (const IntType &o) const { return (Type) v == (Type) o.v; }
  bool operator != (const IntType &o) const { return !(*this == o); }

  IntType& operator += (unsigned count) { *this = *this + count; return *this; }
  IntType& operator -= (unsigned count) { *this = *this - count; return *this; }
  IntType& operator ++ () { *this += 1; return *this; }
  IntType& operator -- () { *this -= 1; return *this; }
  IntType operator ++ (int) { IntType c (*this); ++*this; return c; }
  IntType operator -- (int) { IntType c (*this); --*this; return c; }

  HB_INTERNAL static int cmp (const IntType *a, const IntType *b)
  { return b->cmp (*a); }
  HB_INTERNAL static int cmp (const void *a, const void *b)
  {
    IntType *pa = (IntType *) a;
    IntType *pb = (IntType *) b;

    return pb->cmp (*pa);
  }
  template <typename Type2,
	    hb_enable_if (std::is_integral<Type2>::value &&
			  sizeof (Type2) < sizeof (int) &&
			  sizeof (Type) < sizeof (int))>
  int cmp (Type2 a) const
  {
    Type b = v;
    return (int) a - (int) b;
  }
  template <typename Type2,
	    hb_enable_if (hb_is_convertible (Type2, Type))>
  int cmp (Type2 a) const
  {
    Type b = v;
    return a < b ? -1 : a == b ? 0 : +1;
  }
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }
  protected:
  BEInt<Type, Size> v;
  public:
  DEFINE_SIZE_STATIC (Size);
};

typedef IntType<uint8_t>  HBUINT8;	/* 8-bit unsigned integer. */
typedef IntType<int8_t>   HBINT8;	/* 8-bit signed integer. */
typedef IntType<uint16_t> HBUINT16;	/* 16-bit unsigned integer. */
typedef IntType<int16_t>  HBINT16;	/* 16-bit signed integer. */
typedef IntType<uint32_t> HBUINT32;	/* 32-bit unsigned integer. */
typedef IntType<int32_t>  HBINT32;	/* 32-bit signed integer. */
/* Note: we cannot defined a signed HBINT24 because there's no corresponding C type.
 * Works for unsigned, but not signed, since we rely on compiler for sign-extension. */
typedef IntType<uint32_t, 3> HBUINT24;	/* 24-bit unsigned integer. */

/* 15-bit unsigned number; top bit used for extension. */
struct HBUINT15 : HBUINT16
{
  /* TODO Flesh out; actually mask top bit. */
  HBUINT15& operator = (uint16_t i ) { HBUINT16::operator= (i); return *this; }
  public:
  DEFINE_SIZE_STATIC (2);
};

/* 16-bit signed integer (HBINT16) that describes a quantity in FUnits. */
typedef HBINT16 FWORD;

/* 32-bit signed integer (HBINT32) that describes a quantity in FUnits. */
typedef HBINT32 FWORD32;

/* 16-bit unsigned integer (HBUINT16) that describes a quantity in FUnits. */
typedef HBUINT16 UFWORD;

template <typename Type, unsigned fraction_bits>
struct HBFixed : Type
{
  static constexpr float shift = (float) (1 << fraction_bits);
  static_assert (Type::static_size * 8 > fraction_bits, "");

  operator signed () const = delete;
  operator unsigned () const = delete;
  typename Type::type to_int () const { return Type::v; }
  void set_int (typename Type::type i ) { Type::v = i; }
  float to_float (float offset = 0) const  { return ((int32_t) Type::v + offset) / shift; }
  void set_float (float f) { Type::v = roundf (f * shift); }
  public:
  DEFINE_SIZE_STATIC (Type::static_size);
};

/* 16-bit signed fixed number with the low 14 bits of fraction (2.14). */
using F2DOT14 = HBFixed<HBINT16, 14>;
using F4DOT12 = HBFixed<HBINT16, 12>;
using F6DOT10 = HBFixed<HBINT16, 10>;

/* 32-bit signed fixed-point number (16.16). */
using F16DOT16 = HBFixed<HBINT32, 16>;

/* Date represented in number of seconds since 12:00 midnight, January 1,
 * 1904. The value is represented as a signed 64-bit integer. */
struct LONGDATETIME
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }
  protected:
  HBINT32 major;
  HBUINT32 minor;
  public:
  DEFINE_SIZE_STATIC (8);
};

/* Array of four uint8s (length = 32 bits) used to identify a script, language
 * system, feature, or baseline */
struct Tag : HBUINT32
{
  Tag& operator = (hb_tag_t i) { HBUINT32::operator= (i); return *this; }
  /* What the char* converters return is NOT nul-terminated.  Print using "%.4s" */
  operator const char* () const { return reinterpret_cast<const char *> (this); }
  operator char* ()             { return reinterpret_cast<char *> (this); }
  public:
  DEFINE_SIZE_STATIC (4);
};

/* Glyph index number, same as uint16 (length = 16 bits) */
struct HBGlyphID16 : HBUINT16
{
  HBGlyphID16& operator = (uint16_t i) { HBUINT16::operator= (i); return *this; }
};
struct HBGlyphID24 : HBUINT24
{
  HBGlyphID24& operator = (uint32_t i) { HBUINT24::operator= (i); return *this; }
};

/* Script/language-system/feature index */
struct Index : HBUINT16 {
  static constexpr unsigned NOT_FOUND_INDEX = 0xFFFFu;
  Index& operator = (uint16_t i) { HBUINT16::operator= (i); return *this; }
};
DECLARE_NULL_NAMESPACE_BYTES (OT, Index);

typedef Index NameID;

struct VarIdx : HBUINT32 {
  static constexpr unsigned NO_VARIATION = 0xFFFFFFFFu;
  static_assert (NO_VARIATION == HB_OT_LAYOUT_NO_VARIATIONS_INDEX, "");
  static uint32_t add (uint32_t i, unsigned short v)
  {
    if (i == NO_VARIATION) return i;
    return i + v;
  }
  VarIdx& operator = (uint32_t i) { HBUINT32::operator= (i); return *this; }
};
DECLARE_NULL_NAMESPACE_BYTES (OT, VarIdx);

/* Offset, Null offset = 0 */
template <typename Type, bool has_null=true>
struct Offset : Type
{
  Offset& operator = (typename Type::type i) { Type::operator= (i); return *this; }

  typedef Type type;

  bool is_null () const { return has_null && 0 == *this; }

  public:
  DEFINE_SIZE_STATIC (sizeof (Type));
};

typedef Offset<HBUINT16> Offset16;
typedef Offset<HBUINT24> Offset24;
typedef Offset<HBUINT32> Offset32;


/* CheckSum */
struct CheckSum : HBUINT32
{
  CheckSum& operator = (uint32_t i) { HBUINT32::operator= (i); return *this; }

  /* This is reference implementation from the spec. */
  static uint32_t CalcTableChecksum (const HBUINT32 *Table, uint32_t Length)
  {
    uint32_t Sum = 0L;
    assert (0 == (Length & 3));
    const HBUINT32 *EndPtr = Table + Length / HBUINT32::static_size;

    while (Table < EndPtr)
      Sum += *Table++;
    return Sum;
  }

  /* Note: data should be 4byte aligned and have 4byte padding at the end. */
  void set_for_data (const void *data, unsigned int length)
  { *this = CalcTableChecksum ((const HBUINT32 *) data, length); }

  public:
  DEFINE_SIZE_STATIC (4);
};


/*
 * Version Numbers
 */

template <typename FixedType=HBUINT16>
struct FixedVersion
{
  uint32_t to_int () const { return (major << (sizeof (FixedType) * 8)) + minor; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  FixedType major;
  FixedType minor;
  public:
  DEFINE_SIZE_STATIC (2 * sizeof (FixedType));
};


/*
 * Template subclasses of Offset that do the dereferencing.
 * Use: (base+offset)
 */

template <typename Type, bool has_null>
struct _hb_has_null
{
  static const Type *get_null () { return nullptr; }
  static Type *get_crap ()       { return nullptr; }
};
template <typename Type>
struct _hb_has_null<Type, true>
{
  static const Type *get_null () { return &Null (Type); }
  static       Type *get_crap () { return &Crap (Type); }
};

template <typename Type, typename OffsetType, bool has_null=true>
struct OffsetTo : Offset<OffsetType, has_null>
{
  using target_t = Type;

  // Make sure Type is not unbounded; works only for types that are fully defined at OffsetTo time.
  static_assert (has_null == false ||
		 (hb_has_null_size (Type) || !hb_has_min_size (Type)), "");

  HB_DELETE_COPY_ASSIGN (OffsetTo);
  OffsetTo () = default;

  OffsetTo& operator = (typename OffsetType::type i) { OffsetType::operator= (i); return *this; }

  const Type& operator () (const void *base) const
  {
    if (unlikely (this->is_null ())) return *_hb_has_null<Type, has_null>::get_null ();
    return StructAtOffset<const Type> (base, *this);
  }
  Type& operator () (void *base) const
  {
    if (unlikely (this->is_null ())) return *_hb_has_null<Type, has_null>::get_crap ();
    return StructAtOffset<Type> (base, *this);
  }

  template <typename Base,
	    hb_enable_if (hb_is_convertible (const Base, const void *))>
  friend const Type& operator + (const Base &base, const OffsetTo &offset) { return offset ((const void *) base); }
  template <typename Base,
	    hb_enable_if (hb_is_convertible (const Base, const void *))>
  friend const Type& operator + (const OffsetTo &offset, const Base &base) { return offset ((const void *) base); }
  template <typename Base,
	    hb_enable_if (hb_is_convertible (Base, void *))>
  friend Type& operator + (Base &&base, OffsetTo &offset) { return offset ((void *) base); }
  template <typename Base,
	    hb_enable_if (hb_is_convertible (Base, void *))>
  friend Type& operator + (OffsetTo &offset, Base &&base) { return offset ((void *) base); }


  template <typename ...Ts>
  bool serialize_subset (hb_subset_context_t *c, const OffsetTo& src,
			 const void *src_base, Ts&&... ds)
  {
    *this = 0;
    if (src.is_null ())
      return false;

    auto *s = c->serializer;

    s->push ();

    bool ret = c->dispatch (src_base+src, std::forward<Ts> (ds)...);

    if (ret || !has_null)
      s->add_link (*this, s->pop_pack ());
    else
      s->pop_discard ();

    return ret;
  }


  template <typename ...Ts>
  bool serialize_serialize (hb_serialize_context_t *c, Ts&&... ds)
  {
    *this = 0;

    Type* obj = c->push<Type> ();
    bool ret = obj->serialize (c, std::forward<Ts> (ds)...);

    if (ret)
      c->add_link (*this, c->pop_pack ());
    else
      c->pop_discard ();

    return ret;
  }

  /* TODO: Somehow merge this with previous function into a serialize_dispatch(). */
  /* Workaround clang bug: https://bugs.llvm.org/show_bug.cgi?id=23029
   * Can't compile: whence = hb_serialize_context_t::Head followed by Ts&&...
   */
  template <typename ...Ts>
  bool serialize_copy (hb_serialize_context_t *c, const OffsetTo& src,
		       const void *src_base, unsigned dst_bias,
		       hb_serialize_context_t::whence_t whence,
		       Ts&&... ds)
  {
    *this = 0;
    if (src.is_null ())
      return false;

    c->push ();

    bool ret = c->copy (src_base+src, std::forward<Ts> (ds)...);

    c->add_link (*this, c->pop_pack (), whence, dst_bias);

    return ret;
  }

  bool serialize_copy (hb_serialize_context_t *c, const OffsetTo& src,
		       const void *src_base, unsigned dst_bias = 0)
  { return serialize_copy (c, src, src_base, dst_bias, hb_serialize_context_t::Head); }

  bool sanitize_shallow (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);
    //if (unlikely (this->is_null ())) return_trace (true);
    if (unlikely ((const char *) base + (unsigned) *this < (const char *) base)) return_trace (false);
    return_trace (true);
  }

  template <typename ...Ts>
#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool sanitize (hb_sanitize_context_t *c, const void *base, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    return_trace (sanitize_shallow (c, base) &&
		  (this->is_null () ||
		   c->dispatch (StructAtOffset<Type> (base, *this), std::forward<Ts> (ds)...) ||
		   neuter (c)));
  }

  /* Set the offset to Null */
  bool neuter (hb_sanitize_context_t *c) const
  {
    if (!has_null) return false;
    return c->try_set (this, 0);
  }
  DEFINE_SIZE_STATIC (sizeof (OffsetType));
};
/* Partial specializations. */
template <typename Type, bool has_null=true> using Offset16To = OffsetTo<Type, HBUINT16, has_null>;
template <typename Type, bool has_null=true> using Offset24To = OffsetTo<Type, HBUINT24, has_null>;
template <typename Type, bool has_null=true> using Offset32To = OffsetTo<Type, HBUINT32, has_null>;

template <typename Type, typename OffsetType> using NNOffsetTo = OffsetTo<Type, OffsetType, false>;
template <typename Type> using NNOffset16To = Offset16To<Type, false>;
template <typename Type> using NNOffset24To = Offset24To<Type, false>;
template <typename Type> using NNOffset32To = Offset32To<Type, false>;


/*
 * Array Types
 */

template <typename Type>
struct UnsizedArrayOf
{
  typedef Type item_t;
  static constexpr unsigned item_size = hb_static_size (Type);

  HB_DELETE_CREATE_COPY_ASSIGN (UnsizedArrayOf);

  const Type& operator [] (unsigned int i) const
  {
    return arrayZ[i];
  }
  Type& operator [] (unsigned int i)
  {
    return arrayZ[i];
  }

  static unsigned int get_size (unsigned int len)
  { return len * Type::static_size; }

  template <typename T> operator T * () { return arrayZ; }
  template <typename T> operator const T * () const { return arrayZ; }
  hb_array_t<Type> as_array (unsigned int len)
  { return hb_array (arrayZ, len); }
  hb_array_t<const Type> as_array (unsigned int len) const
  { return hb_array (arrayZ, len); }

  template <typename T>
  Type &lsearch (unsigned int len, const T &x, Type &not_found = Crap (Type))
  { return *as_array (len).lsearch (x, &not_found); }
  template <typename T>
  const Type &lsearch (unsigned int len, const T &x, const Type &not_found = Null (Type)) const
  { return *as_array (len).lsearch (x, &not_found); }
  template <typename T>
  bool lfind (unsigned int len, const T &x, unsigned int *i = nullptr,
	      hb_not_found_t not_found = HB_NOT_FOUND_DONT_STORE,
	      unsigned int to_store = (unsigned int) -1) const
  { return as_array (len).lfind (x, i, not_found, to_store); }

  void qsort (unsigned int len, unsigned int start = 0, unsigned int end = (unsigned int) -1)
  { as_array (len).qsort (start, end); }

  bool serialize (hb_serialize_context_t *c, unsigned int items_len, bool clear = true)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_size (this, get_size (items_len), clear))) return_trace (false);
    return_trace (true);
  }
  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, Type))>
  bool serialize (hb_serialize_context_t *c, Iterator items)
  {
    TRACE_SERIALIZE (this);
    unsigned count = hb_len (items);
    if (unlikely (!serialize (c, count, false))) return_trace (false);
    /* TODO Umm. Just exhaust the iterator instead?  Being extra
     * cautious right now.. */
    for (unsigned i = 0; i < count; i++, ++items)
      arrayZ[i] = *items;
    return_trace (true);
  }

  UnsizedArrayOf* copy (hb_serialize_context_t *c, unsigned count) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->start_embed (this);
    if (unlikely (!as_array (count).copy (c))) return_trace (nullptr);
    return_trace (out);
  }

  template <typename ...Ts>
  HB_ALWAYS_INLINE
  bool sanitize (hb_sanitize_context_t *c, unsigned int count, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c, count))) return_trace (false);
    if (!sizeof... (Ts) && hb_is_trivially_copyable(Type)) return_trace (true);
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!c->dispatch (arrayZ[i], std::forward<Ts> (ds)...)))
	return_trace (false);
    return_trace (true);
  }

  bool sanitize_shallow (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_array (arrayZ, count));
  }

  public:
  Type		arrayZ[HB_VAR_ARRAY];
  public:
  DEFINE_SIZE_UNBOUNDED (0);
};

/* Unsized array of offset's */
template <typename Type, typename OffsetType, bool has_null=true>
using UnsizedArray16OfOffsetTo = UnsizedArrayOf<OffsetTo<Type, OffsetType, has_null>>;

/* Unsized array of offsets relative to the beginning of the array itself. */
template <typename Type, typename OffsetType, bool has_null=true>
struct UnsizedListOfOffset16To : UnsizedArray16OfOffsetTo<Type, OffsetType, has_null>
{
  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    const OffsetTo<Type, OffsetType, has_null> *p = &this->arrayZ[i];
    if (unlikely ((const void *) p < (const void *) this->arrayZ)) return Null (Type); /* Overflowed. */
    _hb_compiler_memory_r_barrier ();
    return this+*p;
  }
  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    const OffsetTo<Type, OffsetType, has_null> *p = &this->arrayZ[i];
    if (unlikely ((const void *) p < (const void *) this->arrayZ)) return Crap (Type); /* Overflowed. */
    _hb_compiler_memory_r_barrier ();
    return this+*p;
  }

  template <typename ...Ts>
  bool sanitize (hb_sanitize_context_t *c, unsigned int count, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    return_trace ((UnsizedArray16OfOffsetTo<Type, OffsetType, has_null>
		   ::sanitize (c, count, this, std::forward<Ts> (ds)...)));
  }
};

/* An array with sorted elements.  Supports binary searching. */
template <typename Type>
struct SortedUnsizedArrayOf : UnsizedArrayOf<Type>
{
  hb_sorted_array_t<Type> as_array (unsigned int len)
  { return hb_sorted_array (this->arrayZ, len); }
  hb_sorted_array_t<const Type> as_array (unsigned int len) const
  { return hb_sorted_array (this->arrayZ, len); }
  operator hb_sorted_array_t<Type> ()             { return as_array (); }
  operator hb_sorted_array_t<const Type> () const { return as_array (); }

  template <typename T>
  Type &bsearch (unsigned int len, const T &x, Type &not_found = Crap (Type))
  { return *as_array (len).bsearch (x, &not_found); }
  template <typename T>
  const Type &bsearch (unsigned int len, const T &x, const Type &not_found = Null (Type)) const
  { return *as_array (len).bsearch (x, &not_found); }
  template <typename T>
  bool bfind (unsigned int len, const T &x, unsigned int *i = nullptr,
	      hb_not_found_t not_found = HB_NOT_FOUND_DONT_STORE,
	      unsigned int to_store = (unsigned int) -1) const
  { return as_array (len).bfind (x, i, not_found, to_store); }
};


/* An array with a number of elements. */
template <typename Type, typename LenType>
struct ArrayOf
{
  typedef Type item_t;
  static constexpr unsigned item_size = hb_static_size (Type);

  HB_DELETE_CREATE_COPY_ASSIGN (ArrayOf);

  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= len)) return Null (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i];
  }
  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= len)) return Crap (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i];
  }

  unsigned int get_size () const
  { return len.static_size + len * Type::static_size; }

  explicit operator bool () const { return len; }

  void pop () { len--; }

  hb_array_t<      Type> as_array ()       { return hb_array (arrayZ, len); }
  hb_array_t<const Type> as_array () const { return hb_array (arrayZ, len); }

  /* Iterator. */
  typedef hb_array_t<const Type>   iter_t;
  typedef hb_array_t<      Type> writer_t;
    iter_t   iter () const { return as_array (); }
  writer_t writer ()       { return as_array (); }
  operator   iter_t () const { return   iter (); }
  operator writer_t ()       { return writer (); }

  /* Faster range-based for loop. */
  const Type *begin () const { return arrayZ; }
  const Type *end () const { return arrayZ + len; }

  template <typename T>
  Type &lsearch (const T &x, Type &not_found = Crap (Type))
  { return *as_array ().lsearch (x, &not_found); }
  template <typename T>
  const Type &lsearch (const T &x, const Type &not_found = Null (Type)) const
  { return *as_array ().lsearch (x, &not_found); }
  template <typename T>
  bool lfind (const T &x, unsigned int *i = nullptr,
	      hb_not_found_t not_found = HB_NOT_FOUND_DONT_STORE,
	      unsigned int to_store = (unsigned int) -1) const
  { return as_array ().lfind (x, i, not_found, to_store); }

  void qsort ()
  { as_array ().qsort (); }

  HB_NODISCARD bool serialize (hb_serialize_context_t *c, unsigned items_len, bool clear = true)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    c->check_assign (len, items_len, HB_SERIALIZE_ERROR_ARRAY_OVERFLOW);
    if (unlikely (!c->extend_size (this, get_size (), clear))) return_trace (false);
    return_trace (true);
  }
  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, Type))>
  HB_NODISCARD bool serialize (hb_serialize_context_t *c, Iterator items)
  {
    TRACE_SERIALIZE (this);
    unsigned count = hb_len (items);
    if (unlikely (!serialize (c, count, false))) return_trace (false);
    /* TODO Umm. Just exhaust the iterator instead?  Being extra
     * cautious right now.. */
    for (unsigned i = 0; i < count; i++, ++items)
      arrayZ[i] = *items;
    return_trace (true);
  }

  Type* serialize_append (hb_serialize_context_t *c)
  {
    TRACE_SERIALIZE (this);
    len++;
    if (unlikely (!len || !c->extend (this)))
    {
      len--;
      return_trace (nullptr);
    }
    return_trace (&arrayZ[len - 1]);
  }

  ArrayOf* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->start_embed (this);
    if (unlikely (!c->extend_min (out))) return_trace (nullptr);
    c->check_assign (out->len, len, HB_SERIALIZE_ERROR_ARRAY_OVERFLOW);
    if (unlikely (!as_array ().copy (c))) return_trace (nullptr);
    return_trace (out);
  }

  template <typename ...Ts>
  HB_ALWAYS_INLINE
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    if (!sizeof... (Ts) && hb_is_trivially_copyable(Type)) return_trace (true);
    unsigned int count = len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!c->dispatch (arrayZ[i], std::forward<Ts> (ds)...)))
	return_trace (false);
    return_trace (true);
  }

  bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (len.sanitize (c) && c->check_array_sized (arrayZ, len, sizeof (LenType)));
  }

  public:
  LenType	len;
  Type		arrayZ[HB_VAR_ARRAY];
  public:
  DEFINE_SIZE_ARRAY (sizeof (LenType), arrayZ);
};
template <typename Type> using Array16Of = ArrayOf<Type, HBUINT16>;
template <typename Type> using Array24Of = ArrayOf<Type, HBUINT24>;
template <typename Type> using Array32Of = ArrayOf<Type, HBUINT32>;
using PString = ArrayOf<HBUINT8, HBUINT8>;

/* Array of Offset's */
template <typename Type> using Array16OfOffset16To = ArrayOf<OffsetTo<Type, HBUINT16>, HBUINT16>;
template <typename Type> using Array16OfOffset32To = ArrayOf<OffsetTo<Type, HBUINT32>, HBUINT16>;
template <typename Type> using Array32OfOffset32To = ArrayOf<OffsetTo<Type, HBUINT32>, HBUINT32>;

/* Array of offsets relative to the beginning of the array itself. */
template <typename Type, typename OffsetType>
struct List16OfOffsetTo : ArrayOf<OffsetTo<Type, OffsetType>, HBUINT16>
{
  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= this->len)) return Null (Type);
    _hb_compiler_memory_r_barrier ();
    return this+this->arrayZ[i];
  }
  const Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= this->len)) return Crap (Type);
    _hb_compiler_memory_r_barrier ();
    return this+this->arrayZ[i];
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    struct List16OfOffsetTo *out = c->serializer->embed (*this);
    if (unlikely (!out)) return_trace (false);
    unsigned int count = this->len;
    for (unsigned int i = 0; i < count; i++)
      out->arrayZ[i].serialize_subset (c, this->arrayZ[i], this, out);
    return_trace (true);
  }

  template <typename ...Ts>
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    return_trace ((Array16Of<OffsetTo<Type, OffsetType>>::sanitize (c, this, std::forward<Ts> (ds)...)));
  }
};

template <typename Type>
using List16OfOffset16To = List16OfOffsetTo<Type, HBUINT16>;

/* An array starting at second element. */
template <typename Type, typename LenType>
struct HeadlessArrayOf
{
  static constexpr unsigned item_size = Type::static_size;

  HB_DELETE_CREATE_COPY_ASSIGN (HeadlessArrayOf);

  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= lenP1 || !i)) return Null (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i-1];
  }
  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= lenP1 || !i)) return Crap (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i-1];
  }
  unsigned int get_size () const
  { return lenP1.static_size + get_length () * Type::static_size; }

  unsigned get_length () const { return lenP1 ? lenP1 - 1 : 0; }

  hb_array_t<      Type> as_array ()       { return hb_array (arrayZ, get_length ()); }
  hb_array_t<const Type> as_array () const { return hb_array (arrayZ, get_length ()); }

  /* Iterator. */
  typedef hb_array_t<const Type>   iter_t;
  typedef hb_array_t<      Type> writer_t;
    iter_t   iter () const { return as_array (); }
  writer_t writer ()       { return as_array (); }
  operator   iter_t () const { return   iter (); }
  operator writer_t ()       { return writer (); }

  /* Faster range-based for loop. */
  const Type *begin () const { return arrayZ; }
  const Type *end () const { return arrayZ + get_length (); }

  HB_NODISCARD bool serialize (hb_serialize_context_t *c, unsigned int items_len, bool clear = true)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);
    c->check_assign (lenP1, items_len + 1, HB_SERIALIZE_ERROR_ARRAY_OVERFLOW);
    if (unlikely (!c->extend_size (this, get_size (), clear))) return_trace (false);
    return_trace (true);
  }
  template <typename Iterator,
	    hb_requires (hb_is_source_of (Iterator, Type))>
  HB_NODISCARD bool serialize (hb_serialize_context_t *c, Iterator items)
  {
    TRACE_SERIALIZE (this);
    unsigned count = hb_len (items);
    if (unlikely (!serialize (c, count, false))) return_trace (false);
    /* TODO Umm. Just exhaust the iterator instead?  Being extra
     * cautious right now.. */
    for (unsigned i = 0; i < count; i++, ++items)
      arrayZ[i] = *items;
    return_trace (true);
  }

  template <typename ...Ts>
  HB_ALWAYS_INLINE
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    if (!sizeof... (Ts) && hb_is_trivially_copyable(Type)) return_trace (true);
    unsigned int count = get_length ();
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!c->dispatch (arrayZ[i], std::forward<Ts> (ds)...)))
	return_trace (false);
    return_trace (true);
  }

  private:
  bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (lenP1.sanitize (c) &&
		  (!lenP1 || c->check_array_sized (arrayZ, lenP1 - 1, sizeof (LenType))));
  }

  public:
  LenType	lenP1;
  Type		arrayZ[HB_VAR_ARRAY];
  public:
  DEFINE_SIZE_ARRAY (sizeof (LenType), arrayZ);
};
template <typename Type> using HeadlessArray16Of = HeadlessArrayOf<Type, HBUINT16>;

/* An array storing length-1. */
template <typename Type, typename LenType=HBUINT16>
struct ArrayOfM1
{
  HB_DELETE_CREATE_COPY_ASSIGN (ArrayOfM1);

  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i > lenM1)) return Null (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i];
  }
  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i > lenM1)) return Crap (Type);
    _hb_compiler_memory_r_barrier ();
    return arrayZ[i];
  }
  unsigned int get_size () const
  { return lenM1.static_size + (lenM1 + 1) * Type::static_size; }

  template <typename ...Ts>
  HB_ALWAYS_INLINE
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    if (!sizeof... (Ts) && hb_is_trivially_copyable(Type)) return_trace (true);
    unsigned int count = lenM1 + 1;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!c->dispatch (arrayZ[i], std::forward<Ts> (ds)...)))
	return_trace (false);
    return_trace (true);
  }

  private:
  bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (lenM1.sanitize (c) &&
		  (c->check_array_sized (arrayZ, lenM1 + 1, sizeof (LenType))));
  }

  public:
  LenType	lenM1;
  Type		arrayZ[HB_VAR_ARRAY];
  public:
  DEFINE_SIZE_ARRAY (sizeof (LenType), arrayZ);
};

/* An array with sorted elements.  Supports binary searching. */
template <typename Type, typename LenType>
struct SortedArrayOf : ArrayOf<Type, LenType>
{
  hb_sorted_array_t<      Type> as_array ()       { return hb_sorted_array (this->arrayZ, this->len); }
  hb_sorted_array_t<const Type> as_array () const { return hb_sorted_array (this->arrayZ, this->len); }

  /* Iterator. */
  typedef hb_sorted_array_t<const Type>   iter_t;
  typedef hb_sorted_array_t<      Type> writer_t;
    iter_t   iter () const { return as_array (); }
  writer_t writer ()       { return as_array (); }
  operator   iter_t () const { return   iter (); }
  operator writer_t ()       { return writer (); }

  /* Faster range-based for loop. */
  const Type *begin () const { return this->arrayZ; }
  const Type *end () const { return this->arrayZ + this->len; }

  bool serialize (hb_serialize_context_t *c, unsigned int items_len)
  {
    TRACE_SERIALIZE (this);
    bool ret = ArrayOf<Type, LenType>::serialize (c, items_len);
    return_trace (ret);
  }
  template <typename Iterator,
	    hb_requires (hb_is_sorted_source_of (Iterator, Type))>
  bool serialize (hb_serialize_context_t *c, Iterator items)
  {
    TRACE_SERIALIZE (this);
    bool ret = ArrayOf<Type, LenType>::serialize (c, items);
    return_trace (ret);
  }

  template <typename T>
  Type &bsearch (const T &x, Type &not_found = Crap (Type))
  { return *as_array ().bsearch (x, &not_found); }
  template <typename T>
  const Type &bsearch (const T &x, const Type &not_found = Null (Type)) const
  { return *as_array ().bsearch (x, &not_found); }
  template <typename T>
  bool bfind (const T &x, unsigned int *i = nullptr,
	      hb_not_found_t not_found = HB_NOT_FOUND_DONT_STORE,
	      unsigned int to_store = (unsigned int) -1) const
  { return as_array ().bfind (x, i, not_found, to_store); }
};

template <typename Type> using SortedArray16Of = SortedArrayOf<Type, HBUINT16>;
template <typename Type> using SortedArray24Of = SortedArrayOf<Type, HBUINT24>;
template <typename Type> using SortedArray32Of = SortedArrayOf<Type, HBUINT32>;

/*
 * Binary-search arrays
 */

template <typename LenType=HBUINT16>
struct BinSearchHeader
{
  operator uint32_t () const { return len; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  BinSearchHeader& operator = (unsigned int v)
  {
    len = v;
    assert (len == v);
    entrySelector = hb_max (1u, hb_bit_storage (v)) - 1;
    searchRange = 16 * (1u << entrySelector);
    rangeShift = v * 16 > searchRange
		 ? 16 * v - searchRange
		 : 0;
    return *this;
  }

  protected:
  LenType	len;
  LenType	searchRange;
  LenType	entrySelector;
  LenType	rangeShift;

  public:
  DEFINE_SIZE_STATIC (8);
};

template <typename Type, typename LenType=HBUINT16>
using BinSearchArrayOf = SortedArrayOf<Type, BinSearchHeader<LenType>>;


struct VarSizedBinSearchHeader
{

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16	unitSize;	/* Size of a lookup unit for this search in bytes. */
  HBUINT16	nUnits;		/* Number of units of the preceding size to be searched. */
  HBUINT16	searchRange;	/* The value of unitSize times the largest power of 2
				 * that is less than or equal to the value of nUnits. */
  HBUINT16	entrySelector;	/* The log base 2 of the largest power of 2 less than
				 * or equal to the value of nUnits. */
  HBUINT16	rangeShift;	/* The value of unitSize times the difference of the
				 * value of nUnits minus the largest power of 2 less
				 * than or equal to the value of nUnits. */
  public:
  DEFINE_SIZE_STATIC (10);
};

template <typename Type>
struct VarSizedBinSearchArrayOf
{
  static constexpr unsigned item_size = Type::static_size;

  HB_DELETE_CREATE_COPY_ASSIGN (VarSizedBinSearchArrayOf);

  bool last_is_terminator () const
  {
    if (unlikely (!header.nUnits)) return false;

    /* Gah.
     *
     * "The number of termination values that need to be included is table-specific.
     * The value that indicates binary search termination is 0xFFFF." */
    const HBUINT16 *words = &StructAtOffset<HBUINT16> (&bytesZ, (header.nUnits - 1) * header.unitSize);
    unsigned int count = Type::TerminationWordCount;
    for (unsigned int i = 0; i < count; i++)
      if (words[i] != 0xFFFFu)
	return false;
    return true;
  }

  const Type& operator [] (int i_) const
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= get_length ())) return Null (Type);
    _hb_compiler_memory_r_barrier ();
    return StructAtOffset<Type> (&bytesZ, i * header.unitSize);
  }
  Type& operator [] (int i_)
  {
    unsigned int i = (unsigned int) i_;
    if (unlikely (i >= get_length ())) return Crap (Type);
    _hb_compiler_memory_r_barrier ();
    return StructAtOffset<Type> (&bytesZ, i * header.unitSize);
  }
  unsigned int get_length () const
  { return header.nUnits - last_is_terminator (); }
  unsigned int get_size () const
  { return header.static_size + header.nUnits * header.unitSize; }

  template <typename ...Ts>
  HB_ALWAYS_INLINE
  bool sanitize (hb_sanitize_context_t *c, Ts&&... ds) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    if (!sizeof... (Ts) && hb_is_trivially_copyable(Type)) return_trace (true);
    unsigned int count = get_length ();
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!(*this)[i].sanitize (c, std::forward<Ts> (ds)...)))
	return_trace (false);
    return_trace (true);
  }

  template <typename T>
  const Type *bsearch (const T &key) const
  {
    unsigned pos;
    return hb_bsearch_impl (&pos,
			    key,
			    (const void *) bytesZ,
			    get_length (),
			    header.unitSize,
			    _hb_cmp_method<T, Type>)
	   ? (const Type *) (((const char *) &bytesZ) + (pos * header.unitSize))
	   : nullptr;
  }

  private:
  bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (header.sanitize (c) &&
		  Type::static_size <= header.unitSize &&
		  c->check_range (bytesZ.arrayZ,
				  header.nUnits,
				  header.unitSize));
  }

  protected:
  VarSizedBinSearchHeader	header;
  UnsizedArrayOf<HBUINT8>	bytesZ;
  public:
  DEFINE_SIZE_ARRAY (10, bytesZ);
};


} /* namespace OT */


#endif /* HB_OPEN_TYPE_HH */
