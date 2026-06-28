#include "hb-subset-table.hh"

#include "hb-ot-var-hvar-table.hh"
#include "hb-ot-var-gvar-table.hh"
#include "hb-ot-var-fvar-table.hh"
#include "hb-ot-var-avar-table.hh"
#include "hb-ot-var-cvar-table.hh"
#include "hb-ot-var-mvar-table.hh"

bool _hb_subset_table_var		(hb_subset_plan_t *plan, hb_vector_t<char> &buf, hb_tag_t tag, bool *success)
{
#ifndef HB_NO_VAR
  switch (tag)
  {
  case HB_TAG('H','V','A','R'): *success = _hb_subset_table<const OT::HVAR> (plan, buf); return true;
  case HB_TAG('V','V','A','R'): *success = _hb_subset_table<const OT::VVAR> (plan, buf); return true;
  case HB_TAG('g','v','a','r'): *success = _hb_subset_table<const OT::gvar> (plan, buf); return true;
  case HB_TAG('f','v','a','r'):
    if (plan->user_axes_location.is_empty ())
      *success = _hb_subset_table_passthrough (plan, tag);
    else
      *success = _hb_subset_table<const OT::fvar> (plan, buf);
    return true;
  case HB_TAG('a','v','a','r'):
    if (plan->user_axes_location.is_empty ())
      *success = _hb_subset_table_passthrough (plan, tag);
    else
      *success = _hb_subset_table<const OT::avar> (plan, buf);
    return true;
  case HB_TAG('c','v','a','r'):
    if (plan->user_axes_location.is_empty ())
      *success = _hb_subset_table_passthrough (plan, tag);
    else
      *success = _hb_subset_table<const OT::cvar> (plan, buf);
    return true;
  case HB_TAG('M','V','A','R'):
    if (plan->user_axes_location.is_empty ())
      *success = _hb_subset_table_passthrough (plan, tag);
    else
      *success = _hb_subset_table<const OT::MVAR> (plan, buf);
    return true;
  }
#endif
  return false;
}
