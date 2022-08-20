#ifndef OT_LAYOUT_GPOS_VALUEFORMAT_HH
#define OT_LAYOUT_GPOS_VALUEFORMAT_HH

#include "../../../hb-ot-layout-gsubgpos.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

typedef HBUINT16 Value;

typedef UnsizedArrayOf<Value> ValueRecord;

struct ValueFormat : HBUINT16
{
  enum Flags {
    xPlacement  = 0x0001u,      /* Includes horizontal adjustment for placement */
    yPlacement  = 0x0002u,      /* Includes vertical adjustment for placement */
    xAdvance    = 0x0004u,      /* Includes horizontal adjustment for advance */
    yAdvance    = 0x0008u,      /* Includes vertical adjustment for advance */
    xPlaDevice  = 0x0010u,      /* Includes horizontal Device table for placement */
    yPlaDevice  = 0x0020u,      /* Includes vertical Device table for placement */
    xAdvDevice  = 0x0040u,      /* Includes horizontal Device table for advance */
    yAdvDevice  = 0x0080u,      /* Includes vertical Device table for advance */
    ignored     = 0x0F00u,      /* Was used in TrueType Open for MM fonts */
    reserved    = 0xF000u,      /* For future use */

    devices     = 0x00F0u       /* Mask for having any Device table */
  };

/* All fields are options.  Only those available advance the value pointer. */
#if 0
  HBINT16               xPlacement;     /* Horizontal adjustment for
                                         * placement--in design units */
  HBINT16               yPlacement;     /* Vertical adjustment for
                                         * placement--in design units */
  HBINT16               xAdvance;       /* Horizontal adjustment for
                                         * advance--in design units (only used
                                         * for horizontal writing) */
  HBINT16               yAdvance;       /* Vertical adjustment for advance--in
                                         * design units (only used for vertical
                                         * writing) */
  Offset16To<Device>    xPlaDevice;     /* Offset to Device table for
                                         * horizontal placement--measured from
                                         * beginning of PosTable (may be NULL) */
  Offset16To<Device>    yPlaDevice;     /* Offset to Device table for vertical
                                         * placement--measured from beginning
                                         * of PosTable (may be NULL) */
  Offset16To<Device>    xAdvDevice;     /* Offset to Device table for
                                         * horizontal advance--measured from
                                         * beginning of PosTable (may be NULL) */
  Offset16To<Device>    yAdvDevice;     /* Offset to Device table for vertical
                                         * advance--measured from beginning of
                                         * PosTable (may be NULL) */
#endif

  IntType& operator = (uint16_t i) { v = i; return *this; }

  unsigned int get_len () const  { return hb_popcount ((unsigned int) *this); }
  unsigned int get_size () const { return get_len () * Value::static_size; }

  bool apply_value (hb_ot_apply_context_t *c,
                    const void            *base,
                    const Value           *values,
                    hb_glyph_position_t   &glyph_pos) const
  {
    bool ret = false;
    unsigned int format = *this;
    if (!format) return ret;

    hb_font_t *font = c->font;
    bool horizontal =
#ifndef HB_NO_VERTICAL
      HB_DIRECTION_IS_HORIZONTAL (c->direction)
#else
      true
#endif
      ;

    if (format & xPlacement) glyph_pos.x_offset  += font->em_scale_x (get_short (values++, &ret));
    if (format & yPlacement) glyph_pos.y_offset  += font->em_scale_y (get_short (values++, &ret));
    if (format & xAdvance) {
      if (likely (horizontal)) glyph_pos.x_advance += font->em_scale_x (get_short (values, &ret));
      values++;
    }
    /* y_advance values grow downward but font-space grows upward, hence negation */
    if (format & yAdvance) {
      if (unlikely (!horizontal)) glyph_pos.y_advance -= font->em_scale_y (get_short (values, &ret));
      values++;
    }

    if (!has_device ()) return ret;

    bool use_x_device = font->x_ppem || font->num_coords;
    bool use_y_device = font->y_ppem || font->num_coords;

    if (!use_x_device && !use_y_device) return ret;

    const VariationStore &store = c->var_store;
    auto *cache = c->var_store_cache;

    /* pixel -> fractional pixel */
    if (format & xPlaDevice) {
      if (use_x_device) glyph_pos.x_offset  += (base + get_device (values, &ret)).get_x_delta (font, store, cache);
      values++;
    }
    if (format & yPlaDevice) {
      if (use_y_device) glyph_pos.y_offset  += (base + get_device (values, &ret)).get_y_delta (font, store, cache);
      values++;
    }
    if (format & xAdvDevice) {
      if (horizontal && use_x_device) glyph_pos.x_advance += (base + get_device (values, &ret)).get_x_delta (font, store, cache);
      values++;
    }
    if (format & yAdvDevice) {
      /* y_advance values grow downward but font-space grows upward, hence negation */
      if (!horizontal && use_y_device) glyph_pos.y_advance -= (base + get_device (values, &ret)).get_y_delta (font, store, cache);
      values++;
    }
    return ret;
  }

  unsigned int get_effective_format (const Value *values) const
  {
    unsigned int format = *this;
    for (unsigned flag = xPlacement; flag <= yAdvDevice; flag = flag << 1) {
      if (format & flag) should_drop (*values++, (Flags) flag, &format);
    }

    return format;
  }

  template<typename Iterator,
      hb_requires (hb_is_iterator (Iterator))>
  unsigned int get_effective_format (Iterator it) const {
    unsigned int new_format = 0;

    for (const hb_array_t<const Value>& values : it)
      new_format = new_format | get_effective_format (&values);

    return new_format;
  }

  void copy_values (hb_serialize_context_t *c,
                    unsigned int new_format,
                    const void *base,
                    const Value *values,
                    const hb_map_t *layout_variation_idx_map) const
  {
    unsigned int format = *this;
    if (!format) return;

    if (format & xPlacement) copy_value (c, new_format, xPlacement, *values++);
    if (format & yPlacement) copy_value (c, new_format, yPlacement, *values++);
    if (format & xAdvance)   copy_value (c, new_format, xAdvance, *values++);
    if (format & yAdvance)   copy_value (c, new_format, yAdvance, *values++);

    if (format & xPlaDevice) copy_device (c, base, values++, layout_variation_idx_map);
    if (format & yPlaDevice) copy_device (c, base, values++, layout_variation_idx_map);
    if (format & xAdvDevice) copy_device (c, base, values++, layout_variation_idx_map);
    if (format & yAdvDevice) copy_device (c, base, values++, layout_variation_idx_map);
  }

  void copy_value (hb_serialize_context_t *c,
                   unsigned int new_format,
                   Flags flag,
                   Value value) const
  {
    // Filter by new format.
    if (!(new_format & flag)) return;
    c->copy (value);
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const void *base,
                                  const hb_array_t<const Value>& values) const
  {
    unsigned format = *this;
    unsigned i = 0;
    if (format & xPlacement) i++;
    if (format & yPlacement) i++;
    if (format & xAdvance) i++;
    if (format & yAdvance) i++;
    if (format & xPlaDevice)
    {
      (base + get_device (&(values[i]))).collect_variation_indices (c->layout_variation_indices);
      i++;
    }

    if (format & ValueFormat::yPlaDevice)
    {
      (base + get_device (&(values[i]))).collect_variation_indices (c->layout_variation_indices);
      i++;
    }

    if (format & ValueFormat::xAdvDevice)
    {

      (base + get_device (&(values[i]))).collect_variation_indices (c->layout_variation_indices);
      i++;
    }

    if (format & ValueFormat::yAdvDevice)
    {

      (base + get_device (&(values[i]))).collect_variation_indices (c->layout_variation_indices);
      i++;
    }
  }

  private:
  bool sanitize_value_devices (hb_sanitize_context_t *c, const void *base, const Value *values) const
  {
    unsigned int format = *this;

    if (format & xPlacement) values++;
    if (format & yPlacement) values++;
    if (format & xAdvance)   values++;
    if (format & yAdvance)   values++;

    if ((format & xPlaDevice) && !get_device (values++).sanitize (c, base)) return false;
    if ((format & yPlaDevice) && !get_device (values++).sanitize (c, base)) return false;
    if ((format & xAdvDevice) && !get_device (values++).sanitize (c, base)) return false;
    if ((format & yAdvDevice) && !get_device (values++).sanitize (c, base)) return false;

    return true;
  }

  static inline Offset16To<Device>& get_device (Value* value)
  {
    return *static_cast<Offset16To<Device> *> (value);
  }
  static inline const Offset16To<Device>& get_device (const Value* value, bool *worked=nullptr)
  {
    if (worked) *worked |= bool (*value);
    return *static_cast<const Offset16To<Device> *> (value);
  }

  bool copy_device (hb_serialize_context_t *c, const void *base,
                    const Value *src_value, const hb_map_t *layout_variation_idx_map) const
  {
    Value       *dst_value = c->copy (*src_value);

    if (!dst_value) return false;
    if (*dst_value == 0) return true;

    *dst_value = 0;
    c->push ();
    if ((base + get_device (src_value)).copy (c, layout_variation_idx_map))
    {
      c->add_link (*dst_value, c->pop_pack ());
      return true;
    }
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  static inline const HBINT16& get_short (const Value* value, bool *worked=nullptr)
  {
    if (worked) *worked |= bool (*value);
    return *reinterpret_cast<const HBINT16 *> (value);
  }

  public:

  bool has_device () const
  {
    unsigned int format = *this;
    return (format & devices) != 0;
  }

  bool sanitize_value (hb_sanitize_context_t *c, const void *base, const Value *values) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_range (values, get_size ()) && (!has_device () || sanitize_value_devices (c, base, values)));
  }

  bool sanitize_values (hb_sanitize_context_t *c, const void *base, const Value *values, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    unsigned int len = get_len ();

    if (!c->check_range (values, count, get_size ())) return_trace (false);

    if (!has_device ()) return_trace (true);

    for (unsigned int i = 0; i < count; i++) {
      if (!sanitize_value_devices (c, base, values))
        return_trace (false);
      values += len;
    }

    return_trace (true);
  }

  /* Just sanitize referenced Device tables.  Doesn't check the values themselves. */
  bool sanitize_values_stride_unsafe (hb_sanitize_context_t *c, const void *base, const Value *values, unsigned int count, unsigned int stride) const
  {
    TRACE_SANITIZE (this);

    if (!has_device ()) return_trace (true);

    for (unsigned int i = 0; i < count; i++) {
      if (!sanitize_value_devices (c, base, values))
        return_trace (false);
      values += stride;
    }

    return_trace (true);
  }

 private:

  void should_drop (Value value, Flags flag, unsigned int* format) const
  {
    if (value) return;
    *format = *format & ~flag;
  }

};

}
}
}

#endif  // #ifndef OT_LAYOUT_GPOS_VALUEFORMAT_HH
