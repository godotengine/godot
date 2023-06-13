#ifndef OT_LAYOUT_GPOS_SINGLEPOS_HH
#define OT_LAYOUT_GPOS_SINGLEPOS_HH

#include "SinglePosFormat1.hh"
#include "SinglePosFormat2.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

struct SinglePos
{
  protected:
  union {
  HBUINT16              format;         /* Format identifier */
  SinglePosFormat1      format1;
  SinglePosFormat2      format2;
  } u;

  public:
  template<typename Iterator,
           hb_requires (hb_is_iterator (Iterator))>
  unsigned get_format (Iterator glyph_val_iter_pairs)
  {
    hb_array_t<const Value> first_val_iter = hb_second (*glyph_val_iter_pairs);

    for (const auto iter : glyph_val_iter_pairs)
      for (const auto _ : hb_zip (iter.second, first_val_iter))
        if (_.first != _.second)
          return 2;

    return 1;
  }

  template<typename Iterator,
      typename SrcLookup,
      hb_requires (hb_is_iterator (Iterator))>
  void serialize (hb_serialize_context_t *c,
                  const SrcLookup* src,
                  Iterator glyph_val_iter_pairs,
                  const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map,
                  bool all_axes_pinned)
  {
    if (unlikely (!c->extend_min (u.format))) return;
    unsigned format = 2;
    ValueFormat new_format = src->get_value_format ();

    if (all_axes_pinned)
      new_format = new_format.drop_device_table_flags ();

    if (glyph_val_iter_pairs)
      format = get_format (glyph_val_iter_pairs);

    u.format = format;
    switch (u.format) {
    case 1: u.format1.serialize (c,
                                 src,
                                 glyph_val_iter_pairs,
                                 new_format,
                                 layout_variation_idx_delta_map);
      return;
    case 2: u.format2.serialize (c,
                                 src,
                                 glyph_val_iter_pairs,
                                 new_format,
                                 layout_variation_idx_delta_map);
      return;
    default:return;
    }
  }

  template <typename context_t, typename ...Ts>
  typename context_t::return_t dispatch (context_t *c, Ts&&... ds) const
  {
    if (unlikely (!c->may_dispatch (this, &u.format))) return c->no_dispatch_return_value ();
    TRACE_DISPATCH (this, u.format);
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1, std::forward<Ts> (ds)...));
    case 2: return_trace (c->dispatch (u.format2, std::forward<Ts> (ds)...));
    default:return_trace (c->default_return_value ());
    }
  }
};


template<typename Iterator, typename SrcLookup>
static void
SinglePos_serialize (hb_serialize_context_t *c,
                     const SrcLookup *src,
                     Iterator it,
                     const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map,
                     bool all_axes_pinned)
{ c->start_embed<SinglePos> ()->serialize (c, src, it, layout_variation_idx_delta_map, all_axes_pinned); }


}
}
}

#endif /* OT_LAYOUT_GPOS_SINGLEPOS_HH */
