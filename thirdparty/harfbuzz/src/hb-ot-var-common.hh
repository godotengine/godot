/*
 * Copyright © 2021  Google, Inc.
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
 */

#ifndef HB_OT_VAR_COMMON_HH
#define HB_OT_VAR_COMMON_HH

#include "hb-ot-layout-common.hh"


namespace OT {

template <typename MapCountT>
struct DeltaSetIndexMapFormat01
{
  friend struct DeltaSetIndexMap;

  private:
  DeltaSetIndexMapFormat01* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    auto *out = c->start_embed (this);
    if (unlikely (!out)) return_trace (nullptr);

    unsigned total_size = min_size + mapCount * get_width ();
    HBUINT8 *p = c->allocate_size<HBUINT8> (total_size);
    if (unlikely (!p)) return_trace (nullptr);

    memcpy (p, this, HBUINT8::static_size * total_size);
    return_trace (out);
  }

  template <typename T>
  bool serialize (hb_serialize_context_t *c, const T &plan)
  {
    unsigned int width = plan.get_width ();
    unsigned int inner_bit_count = plan.get_inner_bit_count ();
    const hb_array_t<const uint32_t> output_map = plan.get_output_map ();

    TRACE_SERIALIZE (this);
    if (unlikely (output_map.length && ((((inner_bit_count-1)&~0xF)!=0) || (((width-1)&~0x3)!=0))))
      return_trace (false);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    entryFormat = ((width-1)<<4)|(inner_bit_count-1);
    mapCount = output_map.length;
    HBUINT8 *p = c->allocate_size<HBUINT8> (width * output_map.length);
    if (unlikely (!p)) return_trace (false);
    for (unsigned int i = 0; i < output_map.length; i++)
    {
      unsigned int v = output_map[i];
      unsigned int outer = v >> 16;
      unsigned int inner = v & 0xFFFF;
      unsigned int u = (outer << inner_bit_count) | inner;
      for (unsigned int w = width; w > 0;)
      {
        p[--w] = u;
        u >>= 8;
      }
      p += width;
    }
    return_trace (true);
  }

  uint32_t map (unsigned int v) const /* Returns 16.16 outer.inner. */
  {
    /* If count is zero, pass value unchanged.  This takes
     * care of direct mapping for advance map. */
    if (!mapCount)
      return v;

    if (v >= mapCount)
      v = mapCount - 1;

    unsigned int u = 0;
    { /* Fetch it. */
      unsigned int w = get_width ();
      const HBUINT8 *p = mapDataZ.arrayZ + w * v;
      for (; w; w--)
        u = (u << 8) + *p++;
    }

    { /* Repack it. */
      unsigned int n = get_inner_bit_count ();
      unsigned int outer = u >> n;
      unsigned int inner = u & ((1 << n) - 1);
      u = (outer<<16) | inner;
    }

    return u;
  }

  unsigned get_map_count () const       { return mapCount; }
  unsigned get_width () const           { return ((entryFormat >> 4) & 3) + 1; }
  unsigned get_inner_bit_count () const { return (entryFormat & 0xF) + 1; }


  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
                  c->check_range (mapDataZ.arrayZ,
                                  mapCount,
                                  get_width ()));
  }

  protected:
  HBUINT8       format;         /* Format identifier--format = 0 */
  HBUINT8       entryFormat;    /* A packed field that describes the compressed
                                 * representation of delta-set indices. */
  MapCountT     mapCount;       /* The number of mapping entries. */
  UnsizedArrayOf<HBUINT8>
                mapDataZ;       /* The delta-set index mapping data. */

  public:
  DEFINE_SIZE_ARRAY (2+MapCountT::static_size, mapDataZ);
};

struct DeltaSetIndexMap
{
  template <typename T>
  bool serialize (hb_serialize_context_t *c, const T &plan)
  {
    TRACE_SERIALIZE (this);
    unsigned length = plan.get_output_map ().length;
    u.format = length <= 0xFFFF ? 0 : 1;
    switch (u.format) {
    case 0: return_trace (u.format0.serialize (c, plan));
    case 1: return_trace (u.format1.serialize (c, plan));
    default:return_trace (false);
    }
  }

  uint32_t map (unsigned v) const
  {
    switch (u.format) {
    case 0: return (u.format0.map (v));
    case 1: return (u.format1.map (v));
    default:return v;
    }
  }

  unsigned get_map_count () const
  {
    switch (u.format) {
    case 0: return u.format0.get_map_count ();
    case 1: return u.format1.get_map_count ();
    default:return 0;
    }
  }

  unsigned get_width () const
  {
    switch (u.format) {
    case 0: return u.format0.get_width ();
    case 1: return u.format1.get_width ();
    default:return 0;
    }
  }

  unsigned get_inner_bit_count () const
  {
    switch (u.format) {
    case 0: return u.format0.get_inner_bit_count ();
    case 1: return u.format1.get_inner_bit_count ();
    default:return 0;
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case 0: return_trace (u.format0.sanitize (c));
    case 1: return_trace (u.format1.sanitize (c));
    default:return_trace (true);
    }
  }

  DeltaSetIndexMap* copy (hb_serialize_context_t *c) const
  {
    TRACE_SERIALIZE (this);
    switch (u.format) {
    case 0: return_trace (reinterpret_cast<DeltaSetIndexMap *> (u.format0.copy (c)));
    case 1: return_trace (reinterpret_cast<DeltaSetIndexMap *> (u.format1.copy (c)));
    default:return_trace (nullptr);
    }
  }

  protected:
  union {
  HBUINT8                            format;         /* Format identifier */
  DeltaSetIndexMapFormat01<HBUINT16> format0;
  DeltaSetIndexMapFormat01<HBUINT32> format1;
  } u;
  public:
  DEFINE_SIZE_UNION (1, format);
};

} /* namespace OT */


#endif /* HB_OT_VAR_COMMON_HH */
