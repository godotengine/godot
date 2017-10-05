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

#ifndef HB_OPEN_TYPE_PRIVATE_HH
#define HB_OPEN_TYPE_PRIVATE_HH

#include "hb-private.hh"
#include "hb-blob-private.hh"
#include "hb-face-private.hh"
#include "hb-machinery-private.hh"


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
template <typename Type, unsigned int Size>
struct IntType
{
  inline void set (Type i) { v.set (i); }
  inline operator Type(void) const { return v; }
  inline bool operator == (const IntType<Type,Size> &o) const { return (Type) v == (Type) o.v; }
  inline bool operator != (const IntType<Type,Size> &o) const { return !(*this == o); }
  static inline int cmp (const IntType<Type,Size> *a, const IntType<Type,Size> *b) { return b->cmp (*a); }
  template <typename Type2>
  inline int cmp (Type2 a) const
  {
    Type b = v;
    if (sizeof (Type) < sizeof (int) && sizeof (Type2) < sizeof (int))
      return (int) a - (int) b;
    else
      return a < b ? -1 : a == b ? 0 : +1;
  }
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }
  protected:
  BEInt<Type, Size> v;
  public:
  DEFINE_SIZE_STATIC (Size);
};

typedef IntType<uint8_t,  1> HBUINT8;	/* 8-bit unsigned integer. */
typedef IntType<int8_t,   1> HBINT8;	/* 8-bit signed integer. */
typedef IntType<uint16_t, 2> HBUINT16;	/* 16-bit unsigned integer. */
typedef IntType<int16_t,  2> HBINT16;	/* 16-bit signed integer. */
typedef IntType<uint32_t, 4> HBUINT32;	/* 32-bit unsigned integer. */
typedef IntType<int32_t,  4> HBINT32;	/* 32-bit signed integer. */
typedef IntType<uint32_t, 3> HBUINT24;	/* 24-bit unsigned integer. */

/* 16-bit signed integer (HBINT16) that describes a quantity in FUnits. */
typedef HBINT16 FWORD;

/* 16-bit unsigned integer (HBUINT16) that describes a quantity in FUnits. */
typedef HBUINT16 UFWORD;

/* 16-bit signed fixed number with the low 14 bits of fraction (2.14). */
struct F2DOT14 : HBINT16
{
  // 16384 means 1<<14
  inline float to_float (void) const { return ((int32_t) v) / 16384.f; }
  inline void set_float (float f) { v.set (round (f * 16384.f)); }
  public:
  DEFINE_SIZE_STATIC (2);
};

/* 32-bit signed fixed-point number (16.16). */
struct Fixed : HBINT32
{
  // 65536 means 1<<16
  inline float to_float (void) const { return ((int32_t) v) / 65536.f; }
  inline void set_float (float f) { v.set (round (f * 65536.f)); }
  public:
  DEFINE_SIZE_STATIC (4);
};

/* Date represented in number of seconds since 12:00 midnight, January 1,
 * 1904. The value is represented as a signed 64-bit integer. */
struct LONGDATETIME
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
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
  /* What the char* converters return is NOT nul-terminated.  Print using "%.4s" */
  inline operator const char* (void) const { return reinterpret_cast<const char *> (&this->v); }
  inline operator char* (void) { return reinterpret_cast<char *> (&this->v); }
  public:
  DEFINE_SIZE_STATIC (4);
};

/* Glyph index number, same as uint16 (length = 16 bits) */
typedef HBUINT16 GlyphID;

/* Name-table index, same as uint16 (length = 16 bits) */
typedef HBUINT16 NameID;

/* Script/language-system/feature index */
struct Index : HBUINT16 {
  static const unsigned int NOT_FOUND_INDEX = 0xFFFFu;
};
DECLARE_NULL_NAMESPACE_BYTES (OT, Index);

/* Offset, Null offset = 0 */
template <typename Type>
struct Offset : Type
{
  inline bool is_null (void) const { return 0 == *this; }

  inline void *serialize (hb_serialize_context_t *c, const void *base)
  {
    void *t = c->start_embed<void> ();
    this->set ((char *) t - (char *) base); /* TODO(serialize) Overflow? */
    return t;
  }

  public:
  DEFINE_SIZE_STATIC (sizeof(Type));
};

typedef Offset<HBUINT16> Offset16;
typedef Offset<HBUINT32> Offset32;


/* CheckSum */
struct CheckSum : HBUINT32
{
  /* This is reference implementation from the spec. */
  static inline uint32_t CalcTableChecksum (const HBUINT32 *Table, uint32_t Length)
  {
    uint32_t Sum = 0L;
    assert (0 == (Length & 3));
    const HBUINT32 *EndPtr = Table + Length / HBUINT32::static_size;

    while (Table < EndPtr)
      Sum += *Table++;
    return Sum;
  }

  /* Note: data should be 4byte aligned and have 4byte padding at the end. */
  inline void set_for_data (const void *data, unsigned int length)
  { set (CalcTableChecksum ((const HBUINT32 *) data, length)); }

  public:
  DEFINE_SIZE_STATIC (4);
};


/*
 * Version Numbers
 */

template <typename FixedType=HBUINT16>
struct FixedVersion
{
  inline uint32_t to_int (void) const { return (major << (sizeof(FixedType) * 8)) + minor; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  FixedType major;
  FixedType minor;
  public:
  DEFINE_SIZE_STATIC (2 * sizeof(FixedType));
};


/*
 * Template subclasses of Offset that do the dereferencing.
 * Use: (base+offset)
 */

template <typename Type, typename OffsetType=HBUINT16>
struct OffsetTo : Offset<OffsetType>
{
  inline const Type& operator () (const void *base) const
  {
    unsigned int offset = *this;
    if (unlikely (!offset)) return Null(Type);
    return StructAtOffset<const Type> (base, offset);
  }
  inline Type& operator () (void *base) const
  {
    unsigned int offset = *this;
    if (unlikely (!offset)) return Crap(Type);
    return StructAtOffset<Type> (base, offset);
  }

  inline Type& serialize (hb_serialize_context_t *c, const void *base)
  {
    return * (Type *) Offset<OffsetType>::serialize (c, base);
  }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);
    unsigned int offset = *this;
    if (unlikely (!offset)) return_trace (true);
    if (unlikely (!c->check_range (base, offset))) return_trace (false);
    const Type &obj = StructAtOffset<Type> (base, offset);
    return_trace (likely (obj.sanitize (c)) || neuter (c));
  }
  template <typename T>
  inline bool sanitize (hb_sanitize_context_t *c, const void *base, T user_data) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this))) return_trace (false);
    unsigned int offset = *this;
    if (unlikely (!offset)) return_trace (true);
    if (unlikely (!c->check_range (base, offset))) return_trace (false);
    const Type &obj = StructAtOffset<Type> (base, offset);
    return_trace (likely (obj.sanitize (c, user_data)) || neuter (c));
  }

  /* Set the offset to Null */
  inline bool neuter (hb_sanitize_context_t *c) const {
    return c->try_set (this, 0);
  }
  DEFINE_SIZE_STATIC (sizeof(OffsetType));
};
template <typename Type> struct LOffsetTo : OffsetTo<Type, HBUINT32> {};
template <typename Base, typename OffsetType, typename Type>
static inline const Type& operator + (const Base &base, const OffsetTo<Type, OffsetType> &offset) { return offset (base); }
template <typename Base, typename OffsetType, typename Type>
static inline Type& operator + (Base &base, OffsetTo<Type, OffsetType> &offset) { return offset (base); }


/*
 * Array Types
 */

/* TODO Use it in ArrayOf, HeadlessArrayOf, and other places around the code base?? */
template <typename Type>
struct UnsizedArrayOf
{
  inline const Type& operator [] (unsigned int i) const { return arrayZ[i]; }
  inline Type& operator [] (unsigned int i) { return arrayZ[i]; }

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c, count))) return_trace (false);

    /* Note: for structs that do not reference other structs,
     * we do not need to call their sanitize() as we already did
     * a bound check on the aggregate array size.  We just include
     * a small unreachable expression to make sure the structs
     * pointed to do have a simple sanitize(), ie. they do not
     * reference other structs via offsets.
     */
    (void) (false && arrayZ[0].sanitize (c));

    return_trace (true);
  }
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count, const void *base) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c, count))) return_trace (false);
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!arrayZ[i].sanitize (c, base)))
        return_trace (false);
    return_trace (true);
  }
  template <typename T>
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count, const void *base, T user_data) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c, count))) return_trace (false);
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!arrayZ[i].sanitize (c, base, user_data)))
        return_trace (false);
    return_trace (true);
  }

  inline bool sanitize_shallow (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_array (arrayZ, arrayZ[0].static_size, count));
  }

  public:
  Type	arrayZ[VAR];
  public:
  DEFINE_SIZE_ARRAY (0, arrayZ);
};

/* Unsized array of offset's */
template <typename Type, typename OffsetType>
struct UnsizedOffsetArrayOf : UnsizedArrayOf<OffsetTo<Type, OffsetType> > {};

/* Unsized array of offsets relative to the beginning of the array itself. */
template <typename Type, typename OffsetType>
struct UnsizedOffsetListOf : UnsizedOffsetArrayOf<Type, OffsetType>
{
  inline const Type& operator [] (unsigned int i) const
  {
    return this+this->arrayZ[i];
  }

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    return_trace ((UnsizedOffsetArrayOf<Type, OffsetType>::sanitize (c, count, this)));
  }
  template <typename T>
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int count, T user_data) const
  {
    TRACE_SANITIZE (this);
    return_trace ((UnsizedOffsetArrayOf<Type, OffsetType>::sanitize (c, count, this, user_data)));
  }
};

/* An array with a number of elements. */
template <typename Type, typename LenType=HBUINT16>
struct ArrayOf
{
  const Type *sub_array (unsigned int start_offset, unsigned int *pcount /* IN/OUT */) const
  {
    unsigned int count = len;
    if (unlikely (start_offset > count))
      count = 0;
    else
      count -= start_offset;
    count = MIN (count, *pcount);
    *pcount = count;
    return arrayZ + start_offset;
  }

  inline const Type& operator [] (unsigned int i) const
  {
    if (unlikely (i >= len)) return Null(Type);
    return arrayZ[i];
  }
  inline Type& operator [] (unsigned int i)
  {
    if (unlikely (i >= len)) return Crap(Type);
    return arrayZ[i];
  }
  inline unsigned int get_size (void) const
  { return len.static_size + len * Type::static_size; }

  inline bool serialize (hb_serialize_context_t *c,
			 unsigned int items_len)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    len.set (items_len); /* TODO(serialize) Overflow? */
    if (unlikely (!c->extend (*this))) return_trace (false);
    return_trace (true);
  }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<Type> &items,
			 unsigned int items_len)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!serialize (c, items_len))) return_trace (false);
    for (unsigned int i = 0; i < items_len; i++)
      arrayZ[i] = items[i];
    items += items_len;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);

    /* Note: for structs that do not reference other structs,
     * we do not need to call their sanitize() as we already did
     * a bound check on the aggregate array size.  We just include
     * a small unreachable expression to make sure the structs
     * pointed to do have a simple sanitize(), ie. they do not
     * reference other structs via offsets.
     */
    (void) (false && arrayZ[0].sanitize (c));

    return_trace (true);
  }
  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    unsigned int count = len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!arrayZ[i].sanitize (c, base)))
        return_trace (false);
    return_trace (true);
  }
  template <typename T>
  inline bool sanitize (hb_sanitize_context_t *c, const void *base, T user_data) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);
    unsigned int count = len;
    for (unsigned int i = 0; i < count; i++)
      if (unlikely (!arrayZ[i].sanitize (c, base, user_data)))
        return_trace (false);
    return_trace (true);
  }

  template <typename SearchType>
  inline int lsearch (const SearchType &x) const
  {
    unsigned int count = len;
    for (unsigned int i = 0; i < count; i++)
      if (!this->arrayZ[i].cmp (x))
        return i;
    return -1;
  }

  inline void qsort (void)
  {
    ::qsort (arrayZ, len, sizeof (Type), Type::cmp);
  }

  private:
  inline bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (len.sanitize (c) && c->check_array (arrayZ, Type::static_size, len));
  }

  public:
  LenType len;
  Type arrayZ[VAR];
  public:
  DEFINE_SIZE_ARRAY (sizeof (LenType), arrayZ);
};
template <typename Type> struct LArrayOf : ArrayOf<Type, HBUINT32> {};
typedef ArrayOf<HBUINT8, HBUINT8> PString;

/* Array of Offset's */
template <typename Type, typename OffsetType=HBUINT16>
struct OffsetArrayOf : ArrayOf<OffsetTo<Type, OffsetType> > {};

/* Array of offsets relative to the beginning of the array itself. */
template <typename Type>
struct OffsetListOf : OffsetArrayOf<Type>
{
  inline const Type& operator [] (unsigned int i) const
  {
    if (unlikely (i >= this->len)) return Null(Type);
    return this+this->arrayZ[i];
  }
  inline const Type& operator [] (unsigned int i)
  {
    if (unlikely (i >= this->len)) return Crap(Type);
    return this+this->arrayZ[i];
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (OffsetArrayOf<Type>::sanitize (c, this));
  }
  template <typename T>
  inline bool sanitize (hb_sanitize_context_t *c, T user_data) const
  {
    TRACE_SANITIZE (this);
    return_trace (OffsetArrayOf<Type>::sanitize (c, this, user_data));
  }
};

/* An array starting at second element. */
template <typename Type, typename LenType=HBUINT16>
struct HeadlessArrayOf
{
  inline const Type& operator [] (unsigned int i) const
  {
    if (unlikely (i >= len || !i)) return Null(Type);
    return arrayZ[i-1];
  }
  inline Type& operator [] (unsigned int i)
  {
    if (unlikely (i >= len || !i)) return Crap(Type);
    return arrayZ[i-1];
  }
  inline unsigned int get_size (void) const
  { return len.static_size + (len ? len - 1 : 0) * Type::static_size; }

  inline bool serialize (hb_serialize_context_t *c,
			 Supplier<Type> &items,
			 unsigned int items_len)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    len.set (items_len); /* TODO(serialize) Overflow? */
    if (unlikely (!items_len)) return_trace (true);
    if (unlikely (!c->extend (*this))) return_trace (false);
    for (unsigned int i = 0; i < items_len - 1; i++)
      arrayZ[i] = items[i];
    items += items_len - 1;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!sanitize_shallow (c))) return_trace (false);

    /* Note: for structs that do not reference other structs,
     * we do not need to call their sanitize() as we already did
     * a bound check on the aggregate array size.  We just include
     * a small unreachable expression to make sure the structs
     * pointed to do have a simple sanitize(), ie. they do not
     * reference other structs via offsets.
     */
    (void) (false && arrayZ[0].sanitize (c));

    return_trace (true);
  }

  private:
  inline bool sanitize_shallow (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (len.sanitize (c) &&
		  (!len || c->check_array (arrayZ, Type::static_size, len - 1)));
  }

  public:
  LenType len;
  Type arrayZ[VAR];
  public:
  DEFINE_SIZE_ARRAY (sizeof (LenType), arrayZ);
};

/* An array with sorted elements.  Supports binary searching. */
template <typename Type, typename LenType=HBUINT16>
struct SortedArrayOf : ArrayOf<Type, LenType>
{
  template <typename SearchType>
  inline int bsearch (const SearchType &x) const
  {
    /* Hand-coded bsearch here since this is in the hot inner loop. */
    const Type *arr = this->arrayZ;
    int min = 0, max = (int) this->len - 1;
    while (min <= max)
    {
      int mid = (min + max) / 2;
      int c = arr[mid].cmp (x);
      if (c < 0)
        max = mid - 1;
      else if (c > 0)
        min = mid + 1;
      else
        return mid;
    }
    return -1;
  }
};

/* Binary-search arrays */
struct BinSearchHeader
{
  inline operator uint32_t (void) const { return len; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  inline void set (unsigned int v)
  {
    len.set (v);
    assert (len == v);
    entrySelector.set (MAX (1u, hb_bit_storage (v)) - 1);
    searchRange.set (16 * (1u << entrySelector));
    rangeShift.set (v * 16 > searchRange
		    ? 16 * v - searchRange
		    : 0);
  }

  protected:
  HBUINT16	len;
  HBUINT16	searchRange;
  HBUINT16	entrySelector;
  HBUINT16	rangeShift;

  public:
  DEFINE_SIZE_STATIC (8);
};

template <typename Type>
struct BinSearchArrayOf : SortedArrayOf<Type, BinSearchHeader> {};


} /* namespace OT */


#endif /* HB_OPEN_TYPE_PRIVATE_HH */
