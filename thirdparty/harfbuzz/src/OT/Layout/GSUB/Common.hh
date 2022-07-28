#ifndef OT_LAYOUT_GSUB_COMMON_HH
#define OT_LAYOUT_GSUB_COMMON_HH

#include "../../../hb-serialize.hh"
#include "../../../hb-ot-layout-gsubgpos.hh"

namespace OT {
namespace Layout {
namespace GSUB_impl {

typedef hb_pair_t<hb_codepoint_t, hb_codepoint_t> hb_codepoint_pair_t;

template<typename Iterator>
static void SingleSubst_serialize (hb_serialize_context_t *c,
                                   Iterator it);

}
}
}

#endif /* OT_LAYOUT_GSUB_COMMON_HH */
