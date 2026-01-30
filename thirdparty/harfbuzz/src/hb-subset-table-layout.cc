#include "hb-subset-table.hh"

#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-layout-gpos-table.hh"
#include "hb-ot-layout-base-table.hh"
#include "hb-ot-math-table.hh"

bool _hb_subset_table_layout		(hb_subset_plan_t *plan, hb_vector_t<char> &buf, hb_tag_t tag, bool *success)
{
#ifndef HB_NO_SUBSET_LAYOUT
  switch (tag)
  {
  case HB_TAG('G','D','E','F'): *success = _hb_subset_table<const OT::GDEF> (plan, buf); return true;
  case HB_TAG('G','S','U','B'): *success = _hb_subset_table<const OT::Layout::GSUB> (plan, buf); return true;
  case HB_TAG('G','P','O','S'): *success = _hb_subset_table<const OT::Layout::GPOS> (plan, buf); return true;
  case HB_TAG('B','A','S','E'): *success = _hb_subset_table<const OT::BASE> (plan, buf); return true;
  case HB_TAG('M','A','T','H'): *success = _hb_subset_table<const OT::MATH> (plan, buf); return true;
  }
#endif
  return false;
}
