#ifndef OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBST_HH
#define OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBST_HH

#include "Common.hh"
#include "ReverseChainSingleSubstFormat1.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct ReverseChainSingleSubst
{
  protected:
  union {
  HBUINT16                              format;         /* Format identifier */
  ReverseChainSingleSubstFormat1        format1;
  } u;

  public:
  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    if (unlikely (!c->may_dispatch (this, &u.format))) return c->no_dispatch_return_value ();
    TRACE_DISPATCH (this, u.format);
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }
};

}
}
}

#endif  /* HB_OT_LAYOUT_GSUB_REVERSECHAINSINGLESUBST_HH */
