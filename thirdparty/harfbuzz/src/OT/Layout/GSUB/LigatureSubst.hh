#ifndef OT_LAYOUT_GSUB_LIGATURESUBST_HH
#define OT_LAYOUT_GSUB_LIGATURESUBST_HH

#include "Common.hh"
#include "LigatureSubstFormat1.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct LigatureSubst
{
  protected:
  union {
  HBUINT16				format;         /* Format identifier */
  LigatureSubstFormat1_2<SmallTypes>	format1;
#ifndef HB_NO_BEYOND_64K
  LigatureSubstFormat1_2<MediumTypes>	format2;
#endif
  } u;

  public:
  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    if (unlikely (!c->may_dispatch (this, &u.format))) return c->no_dispatch_return_value ();
    TRACE_DISPATCH (this, u.format);
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
#ifndef HB_NO_BEYOND_64K
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
#endif
    default:return_trace (c->default_return_value ());
    }
  }

  /* TODO This function is only used by small GIDs, and not updated to 24bit GIDs. Should
   * be done by using iterators. While at it perhaps using iterator of arrays of hb_codepoint_t
   * instead. */
  bool serialize (hb_serialize_context_t *c,
                  hb_sorted_array_t<const HBGlyphID16> first_glyphs,
                  hb_array_t<const unsigned int> ligature_per_first_glyph_count_list,
                  hb_array_t<const HBGlyphID16> ligatures_list,
                  hb_array_t<const unsigned int> component_count_list,
                  hb_array_t<const HBGlyphID16> component_list /* Starting from second for each ligature */)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned int format = 1;
    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c,
                                               first_glyphs,
                                               ligature_per_first_glyph_count_list,
                                               ligatures_list,
                                               component_count_list,
                                               component_list));
    default:return_trace (false);
    }
  }

  /* TODO subset() should choose format. */

};


}
}
}

#endif  /* OT_LAYOUT_GSUB_LIGATURESUBST_HH */
