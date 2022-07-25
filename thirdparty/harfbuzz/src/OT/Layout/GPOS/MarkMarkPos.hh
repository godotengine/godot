#ifndef OT_LAYOUT_GPOS_MARKMARKPOS_HH
#define OT_LAYOUT_GPOS_MARKMARKPOS_HH

#include "MarkMarkPosFormat1.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct MarkMarkPos
{
  protected:
  union {
  HBUINT16				format;         /* Format identifier */
  MarkMarkPosFormat1_2<SmallTypes>	format1;
#ifndef HB_NO_BORING_EXPANSION
  MarkMarkPosFormat1_2<MediumTypes>	format2;
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
#ifndef HB_NO_BORING_EXPANSION
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
#endif
    default:return_trace (c->default_return_value ());
    }
  }
};


}
}
}

#endif /* OT_LAYOUT_GPOS_MARKMARKPOS_HH */
