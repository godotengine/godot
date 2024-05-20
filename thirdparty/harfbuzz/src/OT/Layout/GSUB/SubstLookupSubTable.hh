#ifndef OT_LAYOUT_GSUB_SUBSTLOOKUPSUBTABLE_HH
#define OT_LAYOUT_GSUB_SUBSTLOOKUPSUBTABLE_HH

#include "Common.hh"
#include "SingleSubst.hh"
#include "MultipleSubst.hh"
#include "AlternateSubst.hh"
#include "LigatureSubst.hh"
#include "ContextSubst.hh"
#include "ChainContextSubst.hh"
#include "ExtensionSubst.hh"
#include "ReverseChainSingleSubst.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct SubstLookupSubTable
{
  friend struct ::OT::Lookup;
  friend struct SubstLookup;

  protected:
  union {
  SingleSubst                   single;
  MultipleSubst                 multiple;
  AlternateSubst                alternate;
  LigatureSubst                 ligature;
  ContextSubst                  context;
  ChainContextSubst             chainContext;
  ExtensionSubst                extension;
  ReverseChainSingleSubst       reverseChainContextSingle;
  } u;
  public:
  DEFINE_SIZE_MIN (0);

  enum Type {
    Single              = 1,
    Multiple            = 2,
    Alternate           = 3,
    Ligature            = 4,
    Context             = 5,
    ChainContext        = 6,
    Extension           = 7,
    ReverseChainSingle  = 8
  };

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, unsigned int lookup_type, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, lookup_type);
    switch (lookup_type) {
    case Single:                return_trace (u.single.dispatch (c, std::forward<Ts> (ds)...));
    case Multiple:              return_trace (u.multiple.dispatch (c, std::forward<Ts> (ds)...));
    case Alternate:             return_trace (u.alternate.dispatch (c, std::forward<Ts> (ds)...));
    case Ligature:              return_trace (u.ligature.dispatch (c, std::forward<Ts> (ds)...));
    case Context:               return_trace (u.context.dispatch (c, std::forward<Ts> (ds)...));
    case ChainContext:          return_trace (u.chainContext.dispatch (c, std::forward<Ts> (ds)...));
    case Extension:             return_trace (u.extension.dispatch (c, std::forward<Ts> (ds)...));
    case ReverseChainSingle:    return_trace (u.reverseChainContextSingle.dispatch (c, std::forward<Ts> (ds)...));
    default:                    return_trace (c->default_return_value ());
    }
  }

  bool intersects (const hb_set_t *glyphs, unsigned int lookup_type) const
  {
    hb_intersects_context_t c (glyphs);
    return dispatch (&c, lookup_type);
  }
};


}
}
}

#endif  /* HB_OT_LAYOUT_GSUB_SUBSTLOOKUPSUBTABLE_HH */
