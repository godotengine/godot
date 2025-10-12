#include "hb-subset-table.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-hdmx-table.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-maxp-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-name-table.hh"
#include "hb-ot-post-table.hh"

bool _hb_subset_table_other		(hb_subset_plan_t *plan, hb_vector_t<char> &buf, hb_tag_t tag, bool *success)
{
  switch (tag)
  {
  case HB_TAG('g','l','y','f'): *success = _hb_subset_table<const OT::glyf> (plan, buf); return true;
  case HB_TAG('h','d','m','x'): *success = _hb_subset_table<const OT::hdmx> (plan, buf); return true;
  case HB_TAG('n','a','m','e'): *success = _hb_subset_table<const OT::name> (plan, buf); return true;
  case HB_TAG('h','h','e','a'): *success = true; return true; /* skip hhea, handled by hmtx */
  case HB_TAG('h','m','t','x'): *success = _hb_subset_table<const OT::hmtx> (plan, buf); return true;
  case HB_TAG('v','h','e','a'): *success = true; return true; /* skip vhea, handled by vmtx */
  case HB_TAG('v','m','t','x'): *success = _hb_subset_table<const OT::vmtx> (plan, buf); return true;
  case HB_TAG('m','a','x','p'): *success = _hb_subset_table<const OT::maxp> (plan, buf); return true;
  case HB_TAG('l','o','c','a'): *success = true; return true; /* skip loca, handled by glyf */
  case HB_TAG('c','m','a','p'): *success = _hb_subset_table<const OT::cmap> (plan, buf); return true;
  case HB_TAG('O','S','/','2'): *success = _hb_subset_table<const OT::OS2 > (plan, buf); return true;
  case HB_TAG('p','o','s','t'): *success = _hb_subset_table<const OT::post> (plan, buf); return true;
  }
  return false;
}
