#include "hb-subset-table.hh"

#include "OT/Color/sbix/sbix.hh"
#include "OT/Color/CPAL/CPAL.hh"
#include "OT/Color/COLR/COLR.hh"
#include "OT/Color/CBDT/CBDT.hh"

bool _hb_subset_table_color		(hb_subset_plan_t *plan, hb_vector_t<char> &buf, hb_tag_t tag, bool *success)
{
#ifndef HB_NO_COLOR
  switch (tag)
  {
  case HB_TAG('s','b','i','x'): *success = _hb_subset_table<const OT::sbix> (plan, buf); return true;
  case HB_TAG('C','O','L','R'): *success = _hb_subset_table<const OT::COLR> (plan, buf); return true;
  case HB_TAG('C','P','A','L'): *success = _hb_subset_table<const OT::CPAL> (plan, buf); return true;
  case HB_TAG('C','B','L','C'): *success = _hb_subset_table<const OT::CBLC> (plan, buf); return true;
  case HB_TAG('C','B','D','T'): *success = true; return true; /* skip CBDT, handled by CBLC */
  }
#endif
  return false;
}
