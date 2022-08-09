/*
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2010,2012  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod, Garret Rieger
 */

#ifndef OT_LAYOUT_COMMON_COVERAGEFORMAT2_HH
#define OT_LAYOUT_COMMON_COVERAGEFORMAT2_HH

#include "RangeRecord.hh"

namespace OT {
namespace Layout {
namespace Common {

template <typename Types>
struct CoverageFormat2_4
{
  friend struct Coverage;

  protected:
  HBUINT16      coverageFormat; /* Format identifier--format = 2 */
  SortedArray16Of<RangeRecord<Types>>
                rangeRecord;    /* Array of glyph ranges--ordered by
                                 * Start GlyphID. rangeCount entries
                                 * long */
  public:
  DEFINE_SIZE_ARRAY (4, rangeRecord);

  private:

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (rangeRecord.sanitize (c));
  }

  unsigned int get_coverage (hb_codepoint_t glyph_id) const
  {
    const RangeRecord<Types> &range = rangeRecord.bsearch (glyph_id);
    return likely (range.first <= range.last)
         ? (unsigned int) range.value + (glyph_id - range.first)
         : NOT_COVERED;
  }

  unsigned get_population () const
  {
    typename Types::large_int ret = 0;
    for (const auto &r : rangeRecord)
      ret += r.get_population ();
    return ret > UINT_MAX ? UINT_MAX : (unsigned) ret;
  }

  template <typename Iterator,
      hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c, Iterator glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    /* TODO(iter) Write more efficiently? */

    unsigned num_ranges = 0;
    hb_codepoint_t last = (hb_codepoint_t) -2;
    for (auto g: glyphs)
    {
      if (last + 1 != g)
        num_ranges++;
      last = g;
    }

    if (unlikely (!rangeRecord.serialize (c, num_ranges))) return_trace (false);
    if (!num_ranges) return_trace (true);

    unsigned count = 0;
    unsigned range = (unsigned) -1;
    last = (hb_codepoint_t) -2;
    for (auto g: glyphs)
    {
      if (last + 1 != g)
      {
        range++;
        rangeRecord[range].first = g;
        rangeRecord[range].value = count;
      }
      rangeRecord[range].last = g;
      last = g;
      count++;
    }

    return_trace (true);
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    return hb_any (+ hb_iter (rangeRecord)
                   | hb_map ([glyphs] (const RangeRecord<Types> &range) { return range.intersects (*glyphs); }));
  }
  bool intersects_coverage (const hb_set_t *glyphs, unsigned int index) const
  {
    auto cmp = [] (const void *pk, const void *pr) -> int
    {
      unsigned index = * (const unsigned *) pk;
      const RangeRecord<Types> &range = * (const RangeRecord<Types> *) pr;
      if (index < range.value) return -1;
      if (index > (unsigned int) range.value + (range.last - range.first)) return +1;
      return 0;
    };

    auto arr = rangeRecord.as_array ();
    unsigned idx;
    if (hb_bsearch_impl (&idx, index,
                         arr.arrayZ, arr.length, sizeof (arr[0]),
                         (int (*)(const void *_key, const void *_item)) cmp))
      return arr.arrayZ[idx].intersects (*glyphs);
    return false;
  }

  template <typename IterableOut,
	    hb_requires (hb_is_sink_of (IterableOut, hb_codepoint_t))>
  void intersect_set (const hb_set_t &glyphs, IterableOut &intersect_glyphs) const
  {
    for (const auto& range : rangeRecord)
    {
      hb_codepoint_t last = range.last;
      for (hb_codepoint_t g = range.first - 1;
	   glyphs.next (&g) && g <= last;)
	intersect_glyphs << g;
    }
  }

  template <typename set_t>
  bool collect_coverage (set_t *glyphs) const
  {
    for (const auto& range: rangeRecord)
      if (unlikely (!range.collect_coverage (glyphs)))
        return false;
    return true;
  }

  public:
  /* Older compilers need this to be public. */
  struct iter_t
  {
    void init (const CoverageFormat2_4 &c_)
    {
      c = &c_;
      coverage = 0;
      i = 0;
      j = c->rangeRecord.len ? c->rangeRecord[0].first : 0;
      if (unlikely (c->rangeRecord[0].first > c->rangeRecord[0].last))
      {
        /* Broken table. Skip. */
        i = c->rangeRecord.len;
        j = 0;
      }
    }
    bool __more__ () const { return i < c->rangeRecord.len; }
    void __next__ ()
    {
      if (j >= c->rangeRecord[i].last)
      {
        i++;
        if (__more__ ())
        {
          unsigned int old = coverage;
          j = c->rangeRecord[i].first;
          coverage = c->rangeRecord[i].value;
          if (unlikely (coverage != old + 1))
          {
            /* Broken table. Skip. Important to avoid DoS.
             * Also, our callers depend on coverage being
             * consecutive and monotonically increasing,
             * ie. iota(). */
           i = c->rangeRecord.len;
           j = 0;
           return;
          }
        }
        else
          j = 0;
        return;
      }
      coverage++;
      j++;
    }
    hb_codepoint_t get_glyph () const { return j; }
    bool operator != (const iter_t& o) const
    { return i != o.i || j != o.j; }
    iter_t __end__ () const
    {
      iter_t it;
      it.init (*c);
      it.i = c->rangeRecord.len;
      it.j = 0;
      return it;
    }

    private:
    const struct CoverageFormat2_4 *c;
    unsigned int i, coverage;
    hb_codepoint_t j;
  };
  private:
};

}
}
}

#endif  // #ifndef OT_LAYOUT_COMMON_COVERAGEFORMAT2_HH
