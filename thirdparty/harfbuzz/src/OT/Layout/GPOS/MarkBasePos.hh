#ifndef OT_LAYOUT_GPOS_MARKBASEPOS_HH
#define OT_LAYOUT_GPOS_MARKBASEPOS_HH

#include "MarkBasePosFormat1.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct MarkBasePos
{
  protected:
  union {
  HBUINT16				format;         /* Format identifier */
  MarkBasePosFormat1_2<SmallTypes>	format1;
#ifndef HB_NO_BEYOND_64K
  MarkBasePosFormat1_2<MediumTypes>	format2;
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
};

}
}
}

#endif /* OT_LAYOUT_GPOS_MARKBASEPOS_HH */
