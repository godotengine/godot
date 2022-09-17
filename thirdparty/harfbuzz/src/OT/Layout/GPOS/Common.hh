#ifndef OT_LAYOUT_GPOS_COMMON_HH
#define OT_LAYOUT_GPOS_COMMON_HH

namespace OT {
namespace Layout {
namespace GPOS_impl {

enum attach_type_t {
  ATTACH_TYPE_NONE      = 0X00,

  /* Each attachment should be either a mark or a cursive; can't be both. */
  ATTACH_TYPE_MARK      = 0X01,
  ATTACH_TYPE_CURSIVE   = 0X02,
};

/* buffer **position** var allocations */
#define attach_chain() var.i16[0] /* glyph to which this attaches to, relative to current glyphs; negative for going back, positive for forward. */
#define attach_type() var.u8[2] /* attachment type */
/* Note! if attach_chain() is zero, the value of attach_type() is irrelevant. */

template<typename Iterator, typename SrcLookup>
static void SinglePos_serialize (hb_serialize_context_t *c,
                                 const SrcLookup *src,
                                 Iterator it,
                                 const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map,
                                 bool all_axes_pinned);


}
}
}

#endif  // OT_LAYOUT_GPOS_COMMON_HH
