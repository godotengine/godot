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

#ifndef OT_LAYOUT_COMMON_COVERAGE_HH
#define OT_LAYOUT_COMMON_COVERAGE_HH

#include "../types.hh"
#include "CoverageFormat1.hh"
#include "CoverageFormat2.hh"

namespace OT {
namespace Layout {
namespace Common {

template<typename Iterator>
static inline void Coverage_serialize (hb_serialize_context_t *c,
                                       Iterator it);

struct Coverage
{

  protected:
  union {
  HBUINT16                      format;         /* Format identifier */
  CoverageFormat1_3<SmallTypes> format1;
  CoverageFormat2_4<SmallTypes> format2;
#ifndef HB_NO_BEYOND_64K
  CoverageFormat1_3<MediumTypes>format3;
  CoverageFormat2_4<MediumTypes>format4;
#endif
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);

#ifndef HB_OPTIMIZE_SIZE
  HB_ALWAYS_INLINE
#endif
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    hb_barrier ();
    switch (u.format)
    {
    case 1: return_trace (u.format1.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
#ifndef HB_NO_BEYOND_64K
    case 3: return_trace (u.format3.sanitize (c));
    case 4: return_trace (u.format4.sanitize (c));
#endif
    default:return_trace (true);
    }
  }

  /* Has interface. */
  unsigned operator [] (hb_codepoint_t k) const { return get (k); }
  bool has (hb_codepoint_t k) const { return (*this)[k] != NOT_COVERED; }
  /* Predicate. */
  bool operator () (hb_codepoint_t k) const { return has (k); }

  unsigned int get (hb_codepoint_t k) const { return get_coverage (k); }
  unsigned int get_coverage (hb_codepoint_t glyph_id) const
  {
    switch (u.format) {
    case 1: return u.format1.get_coverage (glyph_id);
    case 2: return u.format2.get_coverage (glyph_id);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.get_coverage (glyph_id);
    case 4: return u.format4.get_coverage (glyph_id);
#endif
    default:return NOT_COVERED;
    }
  }
  unsigned int get_coverage (hb_codepoint_t glyph_id,
			     hb_ot_lookup_cache_t *cache) const
  {
    unsigned coverage;
    if (cache && cache->get (glyph_id, &coverage)) return coverage;
    coverage = get_coverage (glyph_id);
    if (cache) cache->set (glyph_id, coverage);
    return coverage;
  }

  unsigned get_population () const
  {
    switch (u.format) {
    case 1: return u.format1.get_population ();
    case 2: return u.format2.get_population ();
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.get_population ();
    case 4: return u.format4.get_population ();
#endif
    default:return NOT_COVERED;
    }
  }

  template <typename Iterator,
      hb_requires (hb_is_sorted_source_of (Iterator, hb_codepoint_t))>
  bool serialize (hb_serialize_context_t *c, Iterator glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (this))) return_trace (false);

    unsigned count = hb_len (glyphs);
    unsigned num_ranges = 0;
    hb_codepoint_t last = (hb_codepoint_t) -2;
    hb_codepoint_t max = 0;
    bool unsorted = false;
    for (auto g: glyphs)
    {
      if (last != (hb_codepoint_t) -2 && g < last)
	unsorted = true;
      if (last + 1 != g)
	num_ranges++;
      last = g;
      if (g > max) max = g;
    }
    u.format = !unsorted && count <= num_ranges * 3 ? 1 : 2;

#ifndef HB_NO_BEYOND_64K
    if (max > 0xFFFFu)
      u.format += 2;
    if (unlikely (max > 0xFFFFFFu))
#else
    if (unlikely (max > 0xFFFFu))
#endif
    {
      c->check_success (false, HB_SERIALIZE_ERROR_INT_OVERFLOW);
      return_trace (false);
    }

    switch (u.format)
    {
    case 1: return_trace (u.format1.serialize (c, glyphs));
    case 2: return_trace (u.format2.serialize (c, glyphs));
#ifndef HB_NO_BEYOND_64K
    case 3: return_trace (u.format3.serialize (c, glyphs));
    case 4: return_trace (u.format4.serialize (c, glyphs));
#endif
    default:return_trace (false);
    }
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
    auto it =
    + iter ()
    | hb_take (c->plan->source->get_num_glyphs ())
    | hb_map_retains_sorting (c->plan->glyph_map_gsub)
    | hb_filter ([] (hb_codepoint_t glyph) { return glyph != HB_MAP_VALUE_INVALID; })
    ;

    // Cache the iterator result as it will be iterated multiple times
    // by the serialize code below.
    hb_sorted_vector_t<hb_codepoint_t> glyphs (it);
    Coverage_serialize (c->serializer, glyphs.iter ());
    return_trace (bool (glyphs));
  }

  bool intersects (const hb_set_t *glyphs) const
  {
    switch (u.format)
    {
    case 1: return u.format1.intersects (glyphs);
    case 2: return u.format2.intersects (glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersects (glyphs);
    case 4: return u.format4.intersects (glyphs);
#endif
    default:return false;
    }
  }
  bool intersects_coverage (const hb_set_t *glyphs, unsigned int index) const
  {
    switch (u.format)
    {
    case 1: return u.format1.intersects_coverage (glyphs, index);
    case 2: return u.format2.intersects_coverage (glyphs, index);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersects_coverage (glyphs, index);
    case 4: return u.format4.intersects_coverage (glyphs, index);
#endif
    default:return false;
    }
  }

  unsigned cost () const
  {
    switch (u.format) {
    case 1: hb_barrier (); return u.format1.cost ();
    case 2: hb_barrier (); return u.format2.cost ();
#ifndef HB_NO_BEYOND_64K
    case 3: hb_barrier (); return u.format3.cost ();
    case 4: hb_barrier (); return u.format4.cost ();
#endif
    default:return 0u;
    }
  }

  /* Might return false if array looks unsorted.
   * Used for faster rejection of corrupt data. */
  template <typename set_t>
  bool collect_coverage (set_t *glyphs) const
  {
    switch (u.format)
    {
    case 1: return u.format1.collect_coverage (glyphs);
    case 2: return u.format2.collect_coverage (glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.collect_coverage (glyphs);
    case 4: return u.format4.collect_coverage (glyphs);
#endif
    default:return false;
    }
  }

  template <typename IterableOut,
	    hb_requires (hb_is_sink_of (IterableOut, hb_codepoint_t))>
  void intersect_set (const hb_set_t &glyphs, IterableOut&& intersect_glyphs) const
  {
    switch (u.format)
    {
    case 1: return u.format1.intersect_set (glyphs, intersect_glyphs);
    case 2: return u.format2.intersect_set (glyphs, intersect_glyphs);
#ifndef HB_NO_BEYOND_64K
    case 3: return u.format3.intersect_set (glyphs, intersect_glyphs);
    case 4: return u.format4.intersect_set (glyphs, intersect_glyphs);
#endif
    default:return ;
    }
  }

  struct iter_t : hb_iter_with_fallback_t<iter_t, hb_codepoint_t>
  {
    static constexpr bool is_sorted_iterator = true;
    iter_t (const Coverage &c_ = Null (Coverage))
    {
      hb_memset (this, 0, sizeof (*this));
      format = c_.u.format;
      switch (format)
      {
      case 1: u.format1.init (c_.u.format1); return;
      case 2: u.format2.init (c_.u.format2); return;
#ifndef HB_NO_BEYOND_64K
      case 3: u.format3.init (c_.u.format3); return;
      case 4: u.format4.init (c_.u.format4); return;
#endif
      default:                               return;
      }
    }
    bool __more__ () const
    {
      switch (format)
      {
      case 1: return u.format1.__more__ ();
      case 2: return u.format2.__more__ ();
#ifndef HB_NO_BEYOND_64K
      case 3: return u.format3.__more__ ();
      case 4: return u.format4.__more__ ();
#endif
      default:return false;
      }
    }
    void __next__ ()
    {
      switch (format)
      {
      case 1: u.format1.__next__ (); break;
      case 2: u.format2.__next__ (); break;
#ifndef HB_NO_BEYOND_64K
      case 3: u.format3.__next__ (); break;
      case 4: u.format4.__next__ (); break;
#endif
      default:                   break;
      }
    }
    typedef hb_codepoint_t __item_t__;
    __item_t__ __item__ () const { return get_glyph (); }

    hb_codepoint_t get_glyph () const
    {
      switch (format)
      {
      case 1: return u.format1.get_glyph ();
      case 2: return u.format2.get_glyph ();
#ifndef HB_NO_BEYOND_64K
      case 3: return u.format3.get_glyph ();
      case 4: return u.format4.get_glyph ();
#endif
      default:return 0;
      }
    }
    bool operator != (const iter_t& o) const
    {
      if (unlikely (format != o.format)) return true;
      switch (format)
      {
      case 1: return u.format1 != o.u.format1;
      case 2: return u.format2 != o.u.format2;
#ifndef HB_NO_BEYOND_64K
      case 3: return u.format3 != o.u.format3;
      case 4: return u.format4 != o.u.format4;
#endif
      default:return false;
      }
    }
    iter_t __end__ () const
    {
      iter_t it = {};
      it.format = format;
      switch (format)
      {
      case 1: it.u.format1 = u.format1.__end__ (); break;
      case 2: it.u.format2 = u.format2.__end__ (); break;
#ifndef HB_NO_BEYOND_64K
      case 3: it.u.format3 = u.format3.__end__ (); break;
      case 4: it.u.format4 = u.format4.__end__ (); break;
#endif
      default: break;
      }
      return it;
    }

    private:
    unsigned int format;
    union {
#ifndef HB_NO_BEYOND_64K
    CoverageFormat2_4<MediumTypes>::iter_t      format4; /* Put this one first since it's larger; helps shut up compiler. */
    CoverageFormat1_3<MediumTypes>::iter_t      format3;
#endif
    CoverageFormat2_4<SmallTypes>::iter_t       format2; /* Put this one first since it's larger; helps shut up compiler. */
    CoverageFormat1_3<SmallTypes>::iter_t       format1;
    } u;
  };
  iter_t iter () const { return iter_t (*this); }
};

template<typename Iterator>
static inline void
Coverage_serialize (hb_serialize_context_t *c,
                    Iterator it)
{ c->start_embed<Coverage> ()->serialize (c, it); }

}
}
}

#endif  // #ifndef OT_LAYOUT_COMMON_COVERAGE_HH
