#ifndef OT_LAYOUT_GPOS_POSLOOKUPSUBTABLE_HH
#define OT_LAYOUT_GPOS_POSLOOKUPSUBTABLE_HH

#include "SinglePos.hh"
#include "PairPos.hh"
#include "CursivePos.hh"
#include "MarkBasePos.hh"
#include "MarkLigPos.hh"
#include "MarkMarkPos.hh"
#include "ContextPos.hh"
#include "ChainContextPos.hh"
#include "ExtensionPos.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct PosLookupSubTable
{
  friend struct ::OT::Lookup;
  friend struct PosLookup;

  enum Type {
    Single              = 1,
    Pair                = 2,
    Cursive             = 3,
    MarkBase            = 4,
    MarkLig             = 5,
    MarkMark            = 6,
    Context             = 7,
    ChainContext        = 8,
    Extension           = 9
  };

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, unsigned int lookup_type, Ts&&... ds) const
  {
    TRACE_DISPATCH (this, lookup_type);
    switch (lookup_type) {
    case Single:                return_trace (u.single.dispatch (c, std::forward<Ts> (ds)...));
    case Pair:                  return_trace (u.pair.dispatch (c, std::forward<Ts> (ds)...));
    case Cursive:               return_trace (u.cursive.dispatch (c, std::forward<Ts> (ds)...));
    case MarkBase:              return_trace (u.markBase.dispatch (c, std::forward<Ts> (ds)...));
    case MarkLig:               return_trace (u.markLig.dispatch (c, std::forward<Ts> (ds)...));
    case MarkMark:              return_trace (u.markMark.dispatch (c, std::forward<Ts> (ds)...));
    case Context:               return_trace (u.context.dispatch (c, std::forward<Ts> (ds)...));
    case ChainContext:          return_trace (u.chainContext.dispatch (c, std::forward<Ts> (ds)...));
    case Extension:             return_trace (u.extension.dispatch (c, std::forward<Ts> (ds)...));
    default:                    return_trace (c->default_return_value ());
    }
  }

  bool intersects (const hb_set_t *glyphs, unsigned int lookup_type) const
  {
    hb_intersects_context_t c (glyphs);
    return dispatch (&c, lookup_type);
  }

  protected:
  union {
  SinglePos             single;
  PairPos               pair;
  CursivePos            cursive;
  MarkBasePos           markBase;
  MarkLigPos            markLig;
  MarkMarkPos           markMark;
  ContextPos            context;
  ChainContextPos       chainContext;
  ExtensionPos          extension;
  } u;
  public:
  DEFINE_SIZE_MIN (0);
};

}
}
}

#endif  /* HB_OT_LAYOUT_GPOS_POSLOOKUPSUBTABLE_HH */
