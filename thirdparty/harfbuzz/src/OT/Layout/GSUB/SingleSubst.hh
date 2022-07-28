#ifndef OT_LAYOUT_GSUB_SINGLESUBST_HH
#define OT_LAYOUT_GSUB_SINGLESUBST_HH

#include "Common.hh"
#include "SingleSubstFormat1.hh"
#include "SingleSubstFormat2.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct SingleSubst
{
  protected:
  union {
  HBUINT16				format;         /* Format identifier */
  SingleSubstFormat1_3<SmallTypes>	format1;
  SingleSubstFormat2_4<SmallTypes>	format2;
#ifndef HB_NO_BORING_EXPANSION
  SingleSubstFormat1_3<MediumTypes>	format3;
  SingleSubstFormat2_4<MediumTypes>	format4;
#endif
  } u;

  public:

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
#ifndef HB_NO_BORING_EXPANSION
    case 3: return_trace (c->dispatch (u.format3, std::forward<Ts> (ds)...));
    case 4: return_trace (c->dispatch (u.format4, std::forward<Ts> (ds)...));
#endif
    default:return_trace (c->default_return_value ());
    }
  }

  template<typename Iterator,
           hb_requires (hb_is_sorted_source_of (Iterator,
                                                const hb_codepoint_pair_t))>
  bool serialize (hb_serialize_context_t *c,
                  Iterator glyphs)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (u.format))) return_trace (false);
    unsigned format = 2;
    unsigned delta = 0;
    if (glyphs)
    {
      format = 1;
      hb_codepoint_t mask = 0xFFFFu;

#ifndef HB_NO_BORING_EXPANSION
       if (+ glyphs
	   | hb_map_retains_sorting (hb_first)
	   | hb_filter ([] (hb_codepoint_t gid) { return gid > 0xFFFFu; }))
       {
	 format += 2;
	 mask = 0xFFFFFFu;
       }
#endif

      auto get_delta = [=] (hb_codepoint_pair_t _)
                       { return (unsigned) (_.second - _.first) & mask; };
      delta = get_delta (*glyphs);
      if (!hb_all (++(+glyphs), delta, get_delta)) format += 1;
    }

    u.format = format;
    switch (u.format) {
    case 1: return_trace (u.format1.serialize (c,
                                               + glyphs
                                               | hb_map_retains_sorting (hb_first),
                                               delta));
    case 2: return_trace (u.format2.serialize (c, glyphs));
#ifndef HB_NO_BORING_EXPANSION
    case 3: return_trace (u.format3.serialize (c,
                                               + glyphs
                                               | hb_map_retains_sorting (hb_first),
                                               delta));
    case 4: return_trace (u.format4.serialize (c, glyphs));
#endif
    default:return_trace (false);
    }
  }
};

template<typename Iterator>
static void
SingleSubst_serialize (hb_serialize_context_t *c,
                       Iterator it)
{ c->start_embed<SingleSubst> ()->serialize (c, it); }

}
}
}

#endif /* OT_LAYOUT_GSUB_SINGLESUBST_HH */
