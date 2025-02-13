#ifndef OT_LAYOUT_GSUB_EXTENSIONSUBST_HH
#define OT_LAYOUT_GSUB_EXTENSIONSUBST_HH

// TODO(garretrieger): move to new layout.
#include "../../../hb-ot-layout-gsubgpos.hh"
#include "Common.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

struct ExtensionSubst : Extension<ExtensionSubst>
{
  typedef struct SubstLookupSubTable SubTable;
  bool is_reverse () const;
};

}
}
}

#endif  /* OT_LAYOUT_GSUB_EXTENSIONSUBST_HH */
