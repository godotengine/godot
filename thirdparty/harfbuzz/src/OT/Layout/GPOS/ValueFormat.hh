#ifndef OT_LAYOUT_GPOS_VALUEFORMAT_HH
#define OT_LAYOUT_GPOS_VALUEFORMAT_HH

#include "../../../hb-ot-layout-gsubgpos.hh"

namespace OT {
namespace Layout {
namespace GPOS_impl {

typedef HBUINT16 Value;

struct ValueBase {}; // Dummy base class tag for OffsetTo<Value> bases.

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

  hb_vector_t<unsigned> get_device_table_indices () const {
    unsigned i = 0;
    hb_vector_t<unsigned> result;
    unsigned format = *this;

    if (format & xPlacement) i++;
    if (format & yPlacement) i++;
    if (format & xAdvance)   i++;
    if (format & yAdvance)   i++;

    if (format & xPlaDevice) result.push (i++);
    if (format & yPlaDevice) result.push (i++);
    if (format & xAdvDevice) result.push (i++);
    if (format & yAdvDevice) result.push (i++);

    return result;
  }

  bool apply_value (hb_ot_apply_context_t *c,
                    const ValueBase       *base,
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
    if (format & xPlaDevice)
    {
      if (use_x_device) glyph_pos.x_offset  += get_device (values, &ret, base, c->sanitizer).get_x_delta (font, store, cache);
      values++;
    }
    if (format & yPlaDevice)
    {
      if (use_y_device) glyph_pos.y_offset  += get_device (values, &ret, base, c->sanitizer).get_y_delta (font, store, cache);
      values++;
    }
    if (format & xAdvDevice)
    {
      if (horizontal && use_x_device) glyph_pos.x_advance += get_device (values, &ret, base, c->sanitizer).get_x_delta (font, store, cache);
      values++;
    }
    if (format & yAdvDevice)
    {
      /* y_advance values grow downward but font-space grows upward, hence negation */
      if (!horizontal && use_y_device) glyph_pos.y_advance -= get_device (values, &ret, base, c->sanitizer).get_y_delta (font, store, cache);
      values++;
    }
    return ret;
  }

  unsigned int get_effective_format (const Value *values, bool strip_hints, bool strip_empty, const ValueBase *base,
                                     const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *varidx_delta_map) const
  {
    unsigned int format = *this;
    for (unsigned flag = xPlacement; flag <= yAdvDevice; flag = flag << 1) {
      if (format & flag)
      {
        if (strip_hints && flag >= xPlaDevice)
        {
          format = format & ~flag;
          values++;
          continue;
        }
        if (varidx_delta_map && flag >= xPlaDevice)
        {
          update_var_flag (values++, (Flags) flag, &format, base, varidx_delta_map);
          continue;
        }
        /* do not strip empty when instancing, cause we don't know whether the new
         * default value is 0 or not */
        if (strip_empty) should_drop (*values, (Flags) flag, &format);
        values++;
      }
    }

    return format;
  }

  template<typename Iterator,
      hb_requires (hb_is_iterator (Iterator))>
  unsigned int get_effective_format (Iterator it, bool strip_hints, bool strip_empty, const ValueBase *base,
                                     const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *varidx_delta_map) const {
    unsigned int new_format = 0;

    for (const hb_array_t<const Value>& values : it)
      new_format = new_format | get_effective_format (&values, strip_hints, strip_empty, base, varidx_delta_map);

    return new_format;
  }

  void copy_values (hb_serialize_context_t *c,
                    unsigned int new_format,
                    const ValueBase *base,
                    const Value *values,
                    const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map) const
  {
    unsigned int format = *this;
    if (!format) return;

    HBINT16 *x_placement = nullptr, *y_placement = nullptr, *x_adv = nullptr, *y_adv = nullptr;
    if (format & xPlacement) x_placement = copy_value (c, new_format, xPlacement, *values++);
    if (format & yPlacement) y_placement = copy_value (c, new_format, yPlacement, *values++);
    if (format & xAdvance)   x_adv = copy_value (c, new_format, xAdvance, *values++);
    if (format & yAdvance)   y_adv = copy_value (c, new_format, yAdvance, *values++);

    if (!has_device ())
      return;

    if (format & xPlaDevice)
    {
      add_delta_to_value (x_placement, base, values, layout_variation_idx_delta_map);
      copy_device (c, base, values++, layout_variation_idx_delta_map, new_format, xPlaDevice);
    }

    if (format & yPlaDevice)
    {
      add_delta_to_value (y_placement, base, values, layout_variation_idx_delta_map);
      copy_device (c, base, values++, layout_variation_idx_delta_map, new_format, yPlaDevice);
    }

    if (format & xAdvDevice)
    {
      add_delta_to_value (x_adv, base, values, layout_variation_idx_delta_map);
      copy_device (c, base, values++, layout_variation_idx_delta_map, new_format, xAdvDevice);
    }

    if (format & yAdvDevice)
    {
      add_delta_to_value (y_adv, base, values, layout_variation_idx_delta_map);
      copy_device (c, base, values++, layout_variation_idx_delta_map, new_format, yAdvDevice);
    }
  }

  HBINT16* copy_value (hb_serialize_context_t *c,
                       unsigned int new_format,
                       Flags flag,
                       Value value) const
  {
    // Filter by new format.
    if (!(new_format & flag)) return nullptr;
    return reinterpret_cast<HBINT16 *> (c->copy (value));
  }

  void collect_variation_indices (hb_collect_variation_indices_context_t *c,
                                  const ValueBase *base,
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
      (base + get_device (&(values[i]))).collect_variation_indices (c);
      i++;
    }

    if (format & ValueFormat::yPlaDevice)
    {
      (base + get_device (&(values[i]))).collect_variation_indices (c);
      i++;
    }

    if (format & ValueFormat::xAdvDevice)
    {
      (base + get_device (&(values[i]))).collect_variation_indices (c);
      i++;
    }

    if (format & ValueFormat::yAdvDevice)
    {
      (base + get_device (&(values[i]))).collect_variation_indices (c);
      i++;
    }
  }

  private:
  bool sanitize_value_devices (hb_sanitize_context_t *c, const ValueBase *base, const Value *values) const
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

  static inline Offset16To<Device, ValueBase>& get_device (Value* value)
  {
    return *static_cast<Offset16To<Device, ValueBase> *> (value);
  }
  static inline const Offset16To<Device, ValueBase>& get_device (const Value* value)
  {
    return *static_cast<const Offset16To<Device, ValueBase> *> (value);
  }
  static inline const Device& get_device (const Value* value,
					  bool *worked,
					  const ValueBase *base,
					  hb_sanitize_context_t &c)
  {
    if (worked) *worked |= bool (*value);
    auto &offset = *static_cast<const Offset16To<Device> *> (value);

    if (unlikely (!offset.sanitize (&c, base)))
      return Null(Device);
    hb_barrier ();

    return base + offset;
  }

  void add_delta_to_value (HBINT16 *value,
                           const ValueBase *base,
                           const Value *src_value,
                           const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map) const
  {
    if (!value) return;
    unsigned varidx = (base + get_device (src_value)).get_variation_index ();
    hb_pair_t<unsigned, int> *varidx_delta;
    if (!layout_variation_idx_delta_map->has (varidx, &varidx_delta)) return;

    *value += hb_second (*varidx_delta);
  }

  bool copy_device (hb_serialize_context_t *c,
                    const ValueBase *base,
                    const Value *src_value,
                    const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *layout_variation_idx_delta_map,
                    unsigned int new_format, Flags flag) const
  {
    // Filter by new format.
    if (!(new_format & flag)) return true;

    Value       *dst_value = c->copy (*src_value);

    if (!dst_value) return false;
    if (*dst_value == 0) return true;

    *dst_value = 0;
    c->push ();
    if ((base + get_device (src_value)).copy (c, layout_variation_idx_delta_map))
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

  bool sanitize_value (hb_sanitize_context_t *c, const ValueBase *base, const Value *values) const
  {
    TRACE_SANITIZE (this);

    if (unlikely (!c->check_range (values, get_size ()))) return_trace (false);

    if (c->lazy_some_gpos)
      return_trace (true);

    return_trace (!has_device () || sanitize_value_devices (c, base, values));
  }

  bool sanitize_values (hb_sanitize_context_t *c, const ValueBase *base, const Value *values, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    unsigned size = get_size ();

    if (!c->check_range (values, count, size)) return_trace (false);

    if (c->lazy_some_gpos)
      return_trace (true);

    hb_barrier ();
    return_trace (sanitize_values_stride_unsafe (c, base, values, count, size));
  }

  /* Just sanitize referenced Device tables.  Doesn't check the values themselves. */
  bool sanitize_values_stride_unsafe (hb_sanitize_context_t *c, const ValueBase *base, const Value *values, unsigned int count, unsigned int stride) const
  {
    TRACE_SANITIZE (this);

    if (!has_device ()) return_trace (true);

    for (unsigned int i = 0; i < count; i++) {
      if (!sanitize_value_devices (c, base, values))
        return_trace (false);
      values = &StructAtOffset<const Value> (values, stride);
    }

    return_trace (true);
  }

 private:

  void should_drop (Value value, Flags flag, unsigned int* format) const
  {
    if (value) return;
    *format = *format & ~flag;
  }

  void update_var_flag (const Value* value, Flags flag,
                        unsigned int* format, const ValueBase *base,
                        const hb_hashmap_t<unsigned, hb_pair_t<unsigned, int>> *varidx_delta_map) const
  {
    if (*value)
    {
      unsigned varidx = (base + get_device (value)).get_variation_index ();
      hb_pair_t<unsigned, int> *varidx_delta;
      if (varidx_delta_map->has (varidx, &varidx_delta) &&
          varidx_delta->first != HB_OT_LAYOUT_NO_VARIATIONS_INDEX)
        return;
    }
    *format = *format & ~flag;
  }
};

}
}
}

#endif  // #ifndef OT_LAYOUT_GPOS_VALUEFORMAT_HH
