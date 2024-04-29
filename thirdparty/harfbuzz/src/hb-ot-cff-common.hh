/*
 * Copyright © 2018 Adobe Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */
#ifndef HB_OT_CFF_COMMON_HH
#define HB_OT_CFF_COMMON_HH

#include "hb-open-type.hh"
#include "hb-bimap.hh"
#include "hb-ot-layout-common.hh"
#include "hb-cff-interp-dict-common.hh"
#include "hb-subset-plan.hh"

namespace CFF {

using namespace OT;

#define CFF_UNDEF_CODE  0xFFFFFFFF

using objidx_t = hb_serialize_context_t::objidx_t;
using whence_t = hb_serialize_context_t::whence_t;

/* utility macro */
template<typename Type>
static inline const Type& StructAtOffsetOrNull (const void *P, unsigned int offset)
{ return offset ? StructAtOffset<Type> (P, offset) : Null (Type); }

struct code_pair_t
{
  unsigned code;
  hb_codepoint_t glyph;
};


using str_buff_t = hb_vector_t<unsigned char>;
using str_buff_vec_t = hb_vector_t<str_buff_t>;
using glyph_to_sid_map_t = hb_vector_t<code_pair_t>;

struct length_f_t
{
  template <typename Iterable,
	    hb_requires (hb_is_iterable (Iterable))>
  unsigned operator () (const Iterable &_) const { return hb_len (hb_iter (_)); }

  unsigned operator () (unsigned _) const { return _; }
}
HB_FUNCOBJ (length_f);

/* CFF INDEX */
template <typename COUNT>
struct CFFIndex
{
  unsigned int offset_array_size () const
  { return offSize * (count + 1); }

  template <typename Iterable,
	    hb_requires (hb_is_iterable (Iterable))>
  bool serialize (hb_serialize_context_t *c,
		  const Iterable &iterable,
		  const unsigned *p_data_size = nullptr)
  {
    TRACE_SERIALIZE (this);
    unsigned data_size;
    if (p_data_size)
      data_size = *p_data_size;
    else
      total_size (iterable, &data_size);

    auto it = hb_iter (iterable);
    if (unlikely (!serialize_header (c, +it, data_size))) return_trace (false);
    unsigned char *ret = c->allocate_size<unsigned char> (data_size, false);
    if (unlikely (!ret)) return_trace (false);
    for (const auto &_ : +it)
    {
      unsigned len = _.length;
      if (!len)
	continue;
      if (len <= 1)
      {
	*ret++ = *_.arrayZ;
	continue;
      }
      hb_memcpy (ret, _.arrayZ, len);
      ret += len;
    }
    return_trace (true);
  }

  template <typename Iterator,
	    hb_requires (hb_is_iterator (Iterator))>
  bool serialize_header (hb_serialize_context_t *c,
			 Iterator it,
			 unsigned data_size)
  {
    TRACE_SERIALIZE (this);

    unsigned off_size = (hb_bit_storage (data_size + 1) + 7) / 8;

    /* serialize CFFIndex header */
    if (unlikely (!c->extend_min (this))) return_trace (false);
    this->count = hb_len (it);
    if (!this->count) return_trace (true);
    if (unlikely (!c->extend (this->offSize))) return_trace (false);
    this->offSize = off_size;
    if (unlikely (!c->allocate_size<HBUINT8> (off_size * (this->count + 1), false)))
      return_trace (false);

    /* serialize indices */
    unsigned int offset = 1;
    if (HB_OPTIMIZE_SIZE_VAL)
    {
      unsigned int i = 0;
      for (const auto &_ : +it)
      {
	set_offset_at (i++, offset);
	offset += length_f (_);
      }
      set_offset_at (i, offset);
    }
    else
      switch (off_size)
      {
	case 1:
	{
	  HBUINT8 *p = (HBUINT8 *) offsets;
	  for (const auto &_ : +it)
	  {
	    *p++ = offset;
	    offset += length_f (_);
	  }
	  *p = offset;
	}
	break;
	case 2:
	{
	  HBUINT16 *p = (HBUINT16 *) offsets;
	  for (const auto &_ : +it)
	  {
	    *p++ = offset;
	    offset += length_f (_);
	  }
	  *p = offset;
	}
	break;
	case 3:
	{
	  HBUINT24 *p = (HBUINT24 *) offsets;
	  for (const auto &_ : +it)
	  {
	    *p++ = offset;
	    offset += length_f (_);
	  }
	  *p = offset;
	}
	break;
	case 4:
	{
	  HBUINT32 *p = (HBUINT32 *) offsets;
	  for (const auto &_ : +it)
	  {
	    *p++ = offset;
	    offset += length_f (_);
	  }
	  *p = offset;
	}
	break;
	default:
	break;
      }

    assert (offset == data_size + 1);
    return_trace (true);
  }

  template <typename Iterable,
	    hb_requires (hb_is_iterable (Iterable))>
  static unsigned total_size (const Iterable &iterable, unsigned *data_size = nullptr)
  {
    auto it = + hb_iter (iterable);
    if (!it)
    {
      if (data_size) *data_size = 0;
      return min_size;
    }

    unsigned total = 0;
    for (const auto &_ : +it)
      total += length_f (_);

    if (data_size) *data_size = total;

    unsigned off_size = (hb_bit_storage (total + 1) + 7) / 8;

    return min_size + HBUINT8::static_size + (hb_len (it) + 1) * off_size + total;
  }

  void set_offset_at (unsigned int index, unsigned int offset)
  {
    assert (index <= count);

    unsigned int size = offSize;
    const HBUINT8 *p = offsets;
    switch (size)
    {
      case 1: ((HBUINT8  *) p)[index] = offset; break;
      case 2: ((HBUINT16 *) p)[index] = offset; break;
      case 3: ((HBUINT24 *) p)[index] = offset; break;
      case 4: ((HBUINT32 *) p)[index] = offset; break;
      default: return;
    }
  }

  private:
  unsigned int offset_at (unsigned int index) const
  {
    assert (index <= count);

    unsigned int size = offSize;
    const HBUINT8 *p = offsets;
    switch (size)
    {
      case 1: return ((HBUINT8  *) p)[index];
      case 2: return ((HBUINT16 *) p)[index];
      case 3: return ((HBUINT24 *) p)[index];
      case 4: return ((HBUINT32 *) p)[index];
      default: return 0;
    }
  }

  const unsigned char *data_base () const
  { return (const unsigned char *) this + min_size + offSize.static_size - 1 + offset_array_size (); }
  public:

  hb_ubytes_t operator [] (unsigned int index) const
  {
    if (unlikely (index >= count)) return hb_ubytes_t ();
    _hb_compiler_memory_r_barrier ();
    unsigned offset0 = offset_at (index);
    unsigned offset1 = offset_at (index + 1);
    if (unlikely (offset1 < offset0 || offset1 > offset_at (count)))
      return hb_ubytes_t ();
    return hb_ubytes_t (data_base () + offset0, offset1 - offset0);
  }

  unsigned int get_size () const
  {
    if (count)
      return min_size + offSize.static_size + offset_array_size () + (offset_at (count) - 1);
    return min_size;  /* empty CFFIndex contains count only */
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (count == 0 || /* empty INDEX */
			   (count < count + 1u &&
			    c->check_struct (&offSize) && offSize >= 1 && offSize <= 4 &&
			    c->check_array (offsets, offSize, count + 1u) &&
			    c->check_array ((const HBUINT8*) data_base (), 1, offset_at (count))))));
  }

  public:
  COUNT		count;		/* Number of object data. Note there are (count+1) offsets */
  private:
  HBUINT8	offSize;	/* The byte size of each offset in the offsets array. */
  HBUINT8	offsets[HB_VAR_ARRAY];
				/* The array of (count + 1) offsets into objects array (1-base). */
  /* HBUINT8 data[HB_VAR_ARRAY];	Object data */
  public:
  DEFINE_SIZE_MIN (COUNT::static_size);
};

/* Top Dict, Font Dict, Private Dict */
struct Dict : UnsizedByteStr
{
  template <typename DICTVAL, typename OP_SERIALIZER, typename ...Ts>
  bool serialize (hb_serialize_context_t *c,
		  const DICTVAL &dictval,
		  OP_SERIALIZER& opszr,
		  Ts&&... ds)
  {
    TRACE_SERIALIZE (this);
    for (unsigned int i = 0; i < dictval.get_count (); i++)
      if (unlikely (!opszr.serialize (c, dictval[i], std::forward<Ts> (ds)...)))
	return_trace (false);

    return_trace (true);
  }

  template <typename T, typename V>
  static bool serialize_int_op (hb_serialize_context_t *c, op_code_t op, V value, op_code_t intOp)
  {
    if (unlikely ((!serialize_int<T, V> (c, intOp, value))))
      return false;

    TRACE_SERIALIZE (this);
    /* serialize the opcode */
    HBUINT8 *p = c->allocate_size<HBUINT8> (OpCode_Size (op), false);
    if (unlikely (!p)) return_trace (false);
    if (Is_OpCode_ESC (op))
    {
      *p = OpCode_escape;
      op = Unmake_OpCode_ESC (op);
      p++;
    }
    *p = op;
    return_trace (true);
  }

  template <typename V>
  static bool serialize_int4_op (hb_serialize_context_t *c, op_code_t op, V value)
  { return serialize_int_op<HBINT32> (c, op, value, OpCode_longintdict); }

  template <typename V>
  static bool serialize_int2_op (hb_serialize_context_t *c, op_code_t op, V value)
  { return serialize_int_op<HBINT16> (c, op, value, OpCode_shortint); }

  template <typename T, int int_op>
  static bool serialize_link_op (hb_serialize_context_t *c, op_code_t op, objidx_t link, whence_t whence)
  {
    T &ofs = *(T *) (c->head + OpCode_Size (int_op));
    if (unlikely (!serialize_int_op<T> (c, op, 0, int_op))) return false;
    c->add_link (ofs, link, whence);
    return true;
  }

  static bool serialize_link4_op (hb_serialize_context_t *c, op_code_t op, objidx_t link, whence_t whence = whence_t::Head)
  { return serialize_link_op<HBINT32, OpCode_longintdict> (c, op, link, whence); }

  static bool serialize_link2_op (hb_serialize_context_t *c, op_code_t op, objidx_t link, whence_t whence = whence_t::Head)
  { return serialize_link_op<HBINT16, OpCode_shortint> (c, op, link, whence); }
};

struct TopDict : Dict {};
struct FontDict : Dict {};
struct PrivateDict : Dict {};

struct table_info_t
{
  void init () { offset = size = 0; link = 0; }

  unsigned int    offset;
  unsigned int    size;
  objidx_t	  link;
};

template <typename COUNT>
struct FDArray : CFFIndex<COUNT>
{
  template <typename DICTVAL, typename INFO, typename Iterator, typename OP_SERIALIZER>
  bool serialize (hb_serialize_context_t *c,
		  Iterator it,
		  OP_SERIALIZER& opszr)
  {
    TRACE_SERIALIZE (this);

    /* serialize INDEX data */
    hb_vector_t<unsigned> sizes;
    if (it.is_random_access_iterator)
      sizes.alloc (hb_len (it));

    c->push ();
    char *data_base = c->head;
    + it
    | hb_map ([&] (const hb_pair_t<const DICTVAL&, const INFO&> &_)
    {
      FontDict *dict = c->start_embed<FontDict> ();
		dict->serialize (c, _.first, opszr, _.second);
		return c->head - (const char*)dict;
	      })
    | hb_sink (sizes)
    ;
    unsigned data_size = c->head - data_base;
    c->pop_pack (false);

    if (unlikely (sizes.in_error ())) return_trace (false);

    /* It just happens that the above is packed right after the header below.
     * Such a hack. */

    /* serialize INDEX header */
    return_trace (CFFIndex<COUNT>::serialize_header (c, hb_iter (sizes), data_size));
  }
};

/* FDSelect */
struct FDSelect0 {
  bool sanitize (hb_sanitize_context_t *c, unsigned int fdcount) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!(c->check_struct (this))))
      return_trace (false);
    if (unlikely (!c->check_array (fds, c->get_num_glyphs ())))
      return_trace (false);

    return_trace (true);
  }

  unsigned get_fd (hb_codepoint_t glyph) const
  { return fds[glyph]; }

  hb_pair_t<unsigned, hb_codepoint_t> get_fd_range (hb_codepoint_t glyph) const
  { return {fds[glyph], glyph + 1}; }

  unsigned int get_size (unsigned int num_glyphs) const
  { return HBUINT8::static_size * num_glyphs; }

  HBUINT8     fds[HB_VAR_ARRAY];

  DEFINE_SIZE_MIN (0);
};

template <typename GID_TYPE, typename FD_TYPE>
struct FDSelect3_4_Range
{
  bool sanitize (hb_sanitize_context_t *c, const void * /*nullptr*/, unsigned int fdcount) const
  {
    TRACE_SANITIZE (this);
    return_trace (first < c->get_num_glyphs () && (fd < fdcount));
  }

  GID_TYPE    first;
  FD_TYPE     fd;
  public:
  DEFINE_SIZE_STATIC (GID_TYPE::static_size + FD_TYPE::static_size);
};

template <typename GID_TYPE, typename FD_TYPE>
struct FDSelect3_4
{
  unsigned int get_size () const
  { return GID_TYPE::static_size * 2 + ranges.get_size (); }

  bool sanitize (hb_sanitize_context_t *c, unsigned int fdcount) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this) || !ranges.sanitize (c, nullptr, fdcount) ||
		  (nRanges () == 0) || ranges[0].first != 0))
      return_trace (false);

    for (unsigned int i = 1; i < nRanges (); i++)
      if (unlikely (ranges[i - 1].first >= ranges[i].first))
	return_trace (false);

    if (unlikely (!sentinel().sanitize (c) || (sentinel() != c->get_num_glyphs ())))
      return_trace (false);

    return_trace (true);
  }

  static int _cmp_range (const void *_key, const void *_item)
  {
    hb_codepoint_t glyph = * (hb_codepoint_t *) _key;
    FDSelect3_4_Range<GID_TYPE, FD_TYPE> *range = (FDSelect3_4_Range<GID_TYPE, FD_TYPE> *) _item;

    if (glyph < range[0].first) return -1;
    if (glyph < range[1].first) return 0;
    return +1;
  }

  unsigned get_fd (hb_codepoint_t glyph) const
  {
    auto *range = hb_bsearch (glyph, &ranges[0], nRanges () - 1, sizeof (ranges[0]), _cmp_range);
    return range ? range->fd : ranges[nRanges () - 1].fd;
  }

  hb_pair_t<unsigned, hb_codepoint_t> get_fd_range (hb_codepoint_t glyph) const
  {
    auto *range = hb_bsearch (glyph, &ranges[0], nRanges () - 1, sizeof (ranges[0]), _cmp_range);
    unsigned fd = range ? range->fd : ranges[nRanges () - 1].fd;
    hb_codepoint_t end = range ? range[1].first : ranges[nRanges () - 1].first;
    return {fd, end};
  }

  GID_TYPE        &nRanges ()       { return ranges.len; }
  GID_TYPE         nRanges () const { return ranges.len; }
  GID_TYPE       &sentinel ()       { return StructAfter<GID_TYPE> (ranges[nRanges () - 1]); }
  const GID_TYPE &sentinel () const { return StructAfter<GID_TYPE> (ranges[nRanges () - 1]); }

  ArrayOf<FDSelect3_4_Range<GID_TYPE, FD_TYPE>, GID_TYPE> ranges;
  /* GID_TYPE sentinel */

  DEFINE_SIZE_ARRAY (GID_TYPE::static_size, ranges);
};

typedef FDSelect3_4<HBUINT16, HBUINT8> FDSelect3;
typedef FDSelect3_4_Range<HBUINT16, HBUINT8> FDSelect3_Range;

struct FDSelect
{
  bool serialize (hb_serialize_context_t *c, const FDSelect &src, unsigned int num_glyphs)
  {
    TRACE_SERIALIZE (this);
    unsigned int size = src.get_size (num_glyphs);
    FDSelect *dest = c->allocate_size<FDSelect> (size, false);
    if (unlikely (!dest)) return_trace (false);
    hb_memcpy (dest, &src, size);
    return_trace (true);
  }

  unsigned int get_size (unsigned int num_glyphs) const
  {
    switch (format)
    {
    case 0: return format.static_size + u.format0.get_size (num_glyphs);
    case 3: return format.static_size + u.format3.get_size ();
    default:return 0;
    }
  }

  unsigned get_fd (hb_codepoint_t glyph) const
  {
    if (this == &Null (FDSelect)) return 0;

    switch (format)
    {
    case 0: return u.format0.get_fd (glyph);
    case 3: return u.format3.get_fd (glyph);
    default:return 0;
    }
  }
  /* Returns pair of fd and one after last glyph in range. */
  hb_pair_t<unsigned, hb_codepoint_t> get_fd_range (hb_codepoint_t glyph) const
  {
    if (this == &Null (FDSelect)) return {0, 1};

    switch (format)
    {
    case 0: return u.format0.get_fd_range (glyph);
    case 3: return u.format3.get_fd_range (glyph);
    default:return {0, 1};
    }
  }

  bool sanitize (hb_sanitize_context_t *c, unsigned int fdcount) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    switch (format)
    {
    case 0: return_trace (u.format0.sanitize (c, fdcount));
    case 3: return_trace (u.format3.sanitize (c, fdcount));
    default:return_trace (false);
    }
  }

  HBUINT8	format;
  union {
  FDSelect0	format0;
  FDSelect3	format3;
  } u;
  public:
  DEFINE_SIZE_MIN (1);
};

template <typename COUNT>
struct Subrs : CFFIndex<COUNT>
{
  typedef COUNT count_type;
  typedef CFFIndex<COUNT> SUPER;
};

} /* namespace CFF */

#endif /* HB_OT_CFF_COMMON_HH */
