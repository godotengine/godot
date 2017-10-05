/*
 * Copyright © 2007,2008,2009,2010  Red Hat, Inc.
 * Copyright © 2010,2012,2013  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_LAYOUT_GPOS_TABLE_HH
#define HB_OT_LAYOUT_GPOS_TABLE_HH

#include "hb-ot-layout-gsubgpos-private.hh"


namespace OT {


/* buffer **position** var allocations */
#define attach_chain() var.i16[0] /* glyph to which this attaches to, relative to current glyphs; negative for going back, positive for forward. */
#define attach_type() var.u8[2] /* attachment type */
/* Note! if attach_chain() is zero, the value of attach_type() is irrelevant. */

enum attach_type_t {
  ATTACH_TYPE_NONE	= 0X00,

  /* Each attachment should be either a mark or a cursive; can't be both. */
  ATTACH_TYPE_MARK	= 0X01,
  ATTACH_TYPE_CURSIVE	= 0X02,
};


/* Shared Tables: ValueRecord, Anchor Table, and MarkArray */

typedef HBUINT16 Value;

typedef Value ValueRecord[VAR];

struct ValueFormat : HBUINT16
{
  enum Flags {
    xPlacement	= 0x0001u,	/* Includes horizontal adjustment for placement */
    yPlacement	= 0x0002u,	/* Includes vertical adjustment for placement */
    xAdvance	= 0x0004u,	/* Includes horizontal adjustment for advance */
    yAdvance	= 0x0008u,	/* Includes vertical adjustment for advance */
    xPlaDevice	= 0x0010u,	/* Includes horizontal Device table for placement */
    yPlaDevice	= 0x0020u,	/* Includes vertical Device table for placement */
    xAdvDevice	= 0x0040u,	/* Includes horizontal Device table for advance */
    yAdvDevice	= 0x0080u,	/* Includes vertical Device table for advance */
    ignored	= 0x0F00u,	/* Was used in TrueType Open for MM fonts */
    reserved	= 0xF000u,	/* For future use */

    devices	= 0x00F0u	/* Mask for having any Device table */
  };

/* All fields are options.  Only those available advance the value pointer. */
#if 0
  HBINT16		xPlacement;		/* Horizontal adjustment for
					 * placement--in design units */
  HBINT16		yPlacement;		/* Vertical adjustment for
					 * placement--in design units */
  HBINT16		xAdvance;		/* Horizontal adjustment for
					 * advance--in design units (only used
					 * for horizontal writing) */
  HBINT16		yAdvance;		/* Vertical adjustment for advance--in
					 * design units (only used for vertical
					 * writing) */
  Offset	xPlaDevice;		/* Offset to Device table for
					 * horizontal placement--measured from
					 * beginning of PosTable (may be NULL) */
  Offset	yPlaDevice;		/* Offset to Device table for vertical
					 * placement--measured from beginning
					 * of PosTable (may be NULL) */
  Offset	xAdvDevice;		/* Offset to Device table for
					 * horizontal advance--measured from
					 * beginning of PosTable (may be NULL) */
  Offset	yAdvDevice;		/* Offset to Device table for vertical
					 * advance--measured from beginning of
					 * PosTable (may be NULL) */
#endif

  inline unsigned int get_len (void) const
  { return hb_popcount ((unsigned int) *this); }
  inline unsigned int get_size (void) const
  { return get_len () * Value::static_size; }

  void apply_value (hb_ot_apply_context_t   *c,
		    const void           *base,
		    const Value          *values,
		    hb_glyph_position_t  &glyph_pos) const
  {
    unsigned int format = *this;
    if (!format) return;

    hb_font_t *font = c->font;
    hb_bool_t horizontal = HB_DIRECTION_IS_HORIZONTAL (c->direction);

    if (format & xPlacement) glyph_pos.x_offset  += font->em_scale_x (get_short (values++));
    if (format & yPlacement) glyph_pos.y_offset  += font->em_scale_y (get_short (values++));
    if (format & xAdvance) {
      if (likely (horizontal)) glyph_pos.x_advance += font->em_scale_x (get_short (values));
      values++;
    }
    /* y_advance values grow downward but font-space grows upward, hence negation */
    if (format & yAdvance) {
      if (unlikely (!horizontal)) glyph_pos.y_advance -= font->em_scale_y (get_short (values));
      values++;
    }

    if (!has_device ()) return;

    bool use_x_device = font->x_ppem || font->num_coords;
    bool use_y_device = font->y_ppem || font->num_coords;

    if (!use_x_device && !use_y_device) return;

    const VariationStore &store = c->var_store;

    /* pixel -> fractional pixel */
    if (format & xPlaDevice) {
      if (use_x_device) glyph_pos.x_offset  += (base + get_device (values)).get_x_delta (font, store);
      values++;
    }
    if (format & yPlaDevice) {
      if (use_y_device) glyph_pos.y_offset  += (base + get_device (values)).get_y_delta (font, store);
      values++;
    }
    if (format & xAdvDevice) {
      if (horizontal && use_x_device) glyph_pos.x_advance += (base + get_device (values)).get_x_delta (font, store);
      values++;
    }
    if (format & yAdvDevice) {
      /* y_advance values grow downward but font-space grows upward, hence negation */
      if (!horizontal && use_y_device) glyph_pos.y_advance -= (base + get_device (values)).get_y_delta (font, store);
      values++;
    }
  }

  private:
  inline bool sanitize_value_devices (hb_sanitize_context_t *c, const void *base, const Value *values) const
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

  static inline OffsetTo<Device>& get_device (Value* value)
  { return *CastP<OffsetTo<Device> > (value); }
  static inline const OffsetTo<Device>& get_device (const Value* value)
  { return *CastP<OffsetTo<Device> > (value); }

  static inline const HBINT16& get_short (const Value* value)
  { return *CastP<HBINT16> (value); }

  public:

  inline bool has_device (void) const {
    unsigned int format = *this;
    return (format & devices) != 0;
  }

  inline bool sanitize_value (hb_sanitize_context_t *c, const void *base, const Value *values) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_range (values, get_size ()) && (!has_device () || sanitize_value_devices (c, base, values)));
  }

  inline bool sanitize_values (hb_sanitize_context_t *c, const void *base, const Value *values, unsigned int count) const
  {
    TRACE_SANITIZE (this);
    unsigned int len = get_len ();

    if (!c->check_array (values, get_size (), count)) return_trace (false);

    if (!has_device ()) return_trace (true);

    for (unsigned int i = 0; i < count; i++) {
      if (!sanitize_value_devices (c, base, values))
        return_trace (false);
      values += len;
    }

    return_trace (true);
  }

  /* Just sanitize referenced Device tables.  Doesn't check the values themselves. */
  inline bool sanitize_values_stride_unsafe (hb_sanitize_context_t *c, const void *base, const Value *values, unsigned int count, unsigned int stride) const
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
};


struct AnchorFormat1
{
  inline void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id HB_UNUSED,
			  float *x, float *y) const
  {
    hb_font_t *font = c->font;
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  FWORD		xCoordinate;		/* Horizontal value--in design units */
  FWORD		yCoordinate;		/* Vertical value--in design units */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct AnchorFormat2
{
  inline void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id,
			  float *x, float *y) const
  {
    hb_font_t *font = c->font;
    unsigned int x_ppem = font->x_ppem;
    unsigned int y_ppem = font->y_ppem;
    hb_position_t cx = 0, cy = 0;
    hb_bool_t ret;

    ret = (x_ppem || y_ppem) &&
	   font->get_glyph_contour_point_for_origin (glyph_id, anchorPoint, HB_DIRECTION_LTR, &cx, &cy);
    *x = ret && x_ppem ? cx : font->em_fscale_x (xCoordinate);
    *y = ret && y_ppem ? cy : font->em_fscale_y (yCoordinate);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 2 */
  FWORD		xCoordinate;		/* Horizontal value--in design units */
  FWORD		yCoordinate;		/* Vertical value--in design units */
  HBUINT16	anchorPoint;		/* Index to glyph contour point */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct AnchorFormat3
{
  inline void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id HB_UNUSED,
			  float *x, float *y) const
  {
    hb_font_t *font = c->font;
    *x = font->em_fscale_x (xCoordinate);
    *y = font->em_fscale_y (yCoordinate);

    if (font->x_ppem || font->num_coords)
      *x += (this+xDeviceTable).get_x_delta (font, c->var_store);
    if (font->y_ppem || font->num_coords)
      *y += (this+yDeviceTable).get_y_delta (font, c->var_store);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && xDeviceTable.sanitize (c, this) && yDeviceTable.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 3 */
  FWORD		xCoordinate;		/* Horizontal value--in design units */
  FWORD		yCoordinate;		/* Vertical value--in design units */
  OffsetTo<Device>
		xDeviceTable;		/* Offset to Device table for X
					 * coordinate-- from beginning of
					 * Anchor table (may be NULL) */
  OffsetTo<Device>
		yDeviceTable;		/* Offset to Device table for Y
					 * coordinate-- from beginning of
					 * Anchor table (may be NULL) */
  public:
  DEFINE_SIZE_STATIC (10);
};

struct Anchor
{
  inline void get_anchor (hb_ot_apply_context_t *c, hb_codepoint_t glyph_id,
			  float *x, float *y) const
  {
    *x = *y = 0;
    switch (u.format) {
    case 1: u.format1.get_anchor (c, glyph_id, x, y); return;
    case 2: u.format2.get_anchor (c, glyph_id, x, y); return;
    case 3: u.format3.get_anchor (c, glyph_id, x, y); return;
    default:					      return;
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case 1: return_trace (u.format1.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
    case 3: return_trace (u.format3.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  AnchorFormat1		format1;
  AnchorFormat2		format2;
  AnchorFormat3		format3;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};


struct AnchorMatrix
{
  inline const Anchor& get_anchor (unsigned int row, unsigned int col, unsigned int cols, bool *found) const {
    *found = false;
    if (unlikely (row >= rows || col >= cols)) return Null(Anchor);
    *found = !matrixZ[row * cols + col].is_null ();
    return this+matrixZ[row * cols + col];
  }

  inline bool sanitize (hb_sanitize_context_t *c, unsigned int cols) const
  {
    TRACE_SANITIZE (this);
    if (!c->check_struct (this)) return_trace (false);
    if (unlikely (hb_unsigned_mul_overflows (rows, cols))) return_trace (false);
    unsigned int count = rows * cols;
    if (!c->check_array (matrixZ, matrixZ[0].static_size, count)) return_trace (false);
    for (unsigned int i = 0; i < count; i++)
      if (!matrixZ[i].sanitize (c, this)) return_trace (false);
    return_trace (true);
  }

  HBUINT16	rows;			/* Number of rows */
  protected:
  OffsetTo<Anchor>
		matrixZ[VAR];		/* Matrix of offsets to Anchor tables--
					 * from beginning of AnchorMatrix table */
  public:
  DEFINE_SIZE_ARRAY (2, matrixZ);
};


struct MarkRecord
{
  friend struct MarkArray;

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && markAnchor.sanitize (c, base));
  }

  protected:
  HBUINT16	klass;			/* Class defined for this mark */
  OffsetTo<Anchor>
		markAnchor;		/* Offset to Anchor table--from
					 * beginning of MarkArray table */
  public:
  DEFINE_SIZE_STATIC (4);
};

struct MarkArray : ArrayOf<MarkRecord>	/* Array of MarkRecords--in Coverage order */
{
  inline bool apply (hb_ot_apply_context_t *c,
		     unsigned int mark_index, unsigned int glyph_index,
		     const AnchorMatrix &anchors, unsigned int class_count,
		     unsigned int glyph_pos) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    const MarkRecord &record = ArrayOf<MarkRecord>::operator[](mark_index);
    unsigned int mark_class = record.klass;

    const Anchor& mark_anchor = this + record.markAnchor;
    bool found;
    const Anchor& glyph_anchor = anchors.get_anchor (glyph_index, mark_class, class_count, &found);
    /* If this subtable doesn't have an anchor for this base and this class,
     * return false such that the subsequent subtables have a chance at it. */
    if (unlikely (!found)) return_trace (false);

    float mark_x, mark_y, base_x, base_y;

    buffer->unsafe_to_break (glyph_pos, buffer->idx);
    mark_anchor.get_anchor (c, buffer->cur().codepoint, &mark_x, &mark_y);
    glyph_anchor.get_anchor (c, buffer->info[glyph_pos].codepoint, &base_x, &base_y);

    hb_glyph_position_t &o = buffer->cur_pos();
    o.x_offset = round (base_x - mark_x);
    o.y_offset = round (base_y - mark_y);
    o.attach_type() = ATTACH_TYPE_MARK;
    o.attach_chain() = (int) glyph_pos - (int) buffer->idx;
    buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_GPOS_ATTACHMENT;

    buffer->idx++;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (ArrayOf<MarkRecord>::sanitize (c, this));
  }
};


/* Lookups */

struct SinglePosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    valueFormat.apply_value (c, this, values, buffer->cur_pos());

    buffer->idx++;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  coverage.sanitize (c, this) &&
		  valueFormat.sanitize_value (c, this, values));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of subtable */
  ValueFormat	valueFormat;		/* Defines the types of data in the
					 * ValueRecord */
  ValueRecord	values;			/* Defines positioning
					 * value(s)--applied to all glyphs in
					 * the Coverage table */
  public:
  DEFINE_SIZE_ARRAY (6, values);
};

struct SinglePosFormat2
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    if (likely (index >= valueCount)) return_trace (false);

    valueFormat.apply_value (c, this,
			     &values[index * valueFormat.get_len ()],
			     buffer->cur_pos());

    buffer->idx++;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  coverage.sanitize (c, this) &&
		  valueFormat.sanitize_values (c, this, values, valueCount));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 2 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of subtable */
  ValueFormat	valueFormat;		/* Defines the types of data in the
					 * ValueRecord */
  HBUINT16	valueCount;		/* Number of ValueRecords */
  ValueRecord	values;			/* Array of ValueRecords--positioning
					 * values applied to glyphs */
  public:
  DEFINE_SIZE_ARRAY (8, values);
};

struct SinglePos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    case 2: return_trace (c->dispatch (u.format2));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  SinglePosFormat1	format1;
  SinglePosFormat2	format2;
  } u;
};


struct PairValueRecord
{
  friend struct PairSet;

  protected:
  GlyphID	secondGlyph;		/* GlyphID of second glyph in the
					 * pair--first glyph is listed in the
					 * Coverage table */
  ValueRecord	values;			/* Positioning data for the first glyph
					 * followed by for second glyph */
  public:
  DEFINE_SIZE_ARRAY (2, values);
};

struct PairSet
{
  friend struct PairPosFormat1;

  inline void collect_glyphs (hb_collect_glyphs_context_t *c,
			      const ValueFormat *valueFormats) const
  {
    TRACE_COLLECT_GLYPHS (this);
    unsigned int len1 = valueFormats[0].get_len ();
    unsigned int len2 = valueFormats[1].get_len ();
    unsigned int record_size = HBUINT16::static_size * (1 + len1 + len2);

    const PairValueRecord *record = CastP<PairValueRecord> (arrayZ);
    c->input->add_array (&record->secondGlyph, len, record_size);
  }

  inline bool apply (hb_ot_apply_context_t *c,
		     const ValueFormat *valueFormats,
		     unsigned int pos) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int len1 = valueFormats[0].get_len ();
    unsigned int len2 = valueFormats[1].get_len ();
    unsigned int record_size = HBUINT16::static_size * (1 + len1 + len2);

    const PairValueRecord *record_array = CastP<PairValueRecord> (arrayZ);
    unsigned int count = len;

    /* Hand-coded bsearch. */
    if (unlikely (!count))
      return_trace (false);
    hb_codepoint_t x = buffer->info[pos].codepoint;
    int min = 0, max = (int) count - 1;
    while (min <= max)
    {
      int mid = (min + max) / 2;
      const PairValueRecord *record = &StructAtOffset<PairValueRecord> (record_array, record_size * mid);
      hb_codepoint_t mid_x = record->secondGlyph;
      if (x < mid_x)
        max = mid - 1;
      else if (x > mid_x)
        min = mid + 1;
      else
      {
        buffer->unsafe_to_break (buffer->idx, pos + 1);
	valueFormats[0].apply_value (c, this, &record->values[0], buffer->cur_pos());
	valueFormats[1].apply_value (c, this, &record->values[len1], buffer->pos[pos]);
	if (len2)
	  pos++;
	buffer->idx = pos;
	return_trace (true);
      }
    }

    return_trace (false);
  }

  struct sanitize_closure_t {
    const void *base;
    const ValueFormat *valueFormats;
    unsigned int len1; /* valueFormats[0].get_len() */
    unsigned int stride; /* 1 + len1 + len2 */
  };

  inline bool sanitize (hb_sanitize_context_t *c, const sanitize_closure_t *closure) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this)
       && c->check_array (arrayZ, HBUINT16::static_size * closure->stride, len))) return_trace (false);

    unsigned int count = len;
    const PairValueRecord *record = CastP<PairValueRecord> (arrayZ);
    return_trace (closure->valueFormats[0].sanitize_values_stride_unsafe (c, closure->base, &record->values[0], count, closure->stride) &&
		  closure->valueFormats[1].sanitize_values_stride_unsafe (c, closure->base, &record->values[closure->len1], count, closure->stride));
  }

  protected:
  HBUINT16	len;			/* Number of PairValueRecords */
  HBUINT16	arrayZ[VAR];		/* Array of PairValueRecords--ordered
					 * by GlyphID of the second glyph */
  public:
  DEFINE_SIZE_ARRAY (2, arrayZ);
};

struct PairPosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    unsigned int count = pairSet.len;
    for (unsigned int i = 0; i < count; i++)
      (this+pairSet[i]).collect_glyphs (c, valueFormat);
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    if (!skippy_iter.next ()) return_trace (false);

    return_trace ((this+pairSet[index]).apply (c, valueFormat, skippy_iter.idx));
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);

    if (!c->check_struct (this)) return_trace (false);

    unsigned int len1 = valueFormat[0].get_len ();
    unsigned int len2 = valueFormat[1].get_len ();
    PairSet::sanitize_closure_t closure = {
      this,
      valueFormat,
      len1,
      1 + len1 + len2
    };

    return_trace (coverage.sanitize (c, this) && pairSet.sanitize (c, this, &closure));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of subtable */
  ValueFormat	valueFormat[2];		/* [0] Defines the types of data in
					 * ValueRecord1--for the first glyph
					 * in the pair--may be zero (0) */
					/* [1] Defines the types of data in
					 * ValueRecord2--for the second glyph
					 * in the pair--may be zero (0) */
  OffsetArrayOf<PairSet>
		pairSet;		/* Array of PairSet tables
					 * ordered by Coverage Index */
  public:
  DEFINE_SIZE_ARRAY (10, pairSet);
};

struct PairPosFormat2
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
    if (unlikely (!(this+classDef2).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int index = (this+coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (index == NOT_COVERED)) return_trace (false);

    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    if (!skippy_iter.next ()) return_trace (false);

    unsigned int len1 = valueFormat1.get_len ();
    unsigned int len2 = valueFormat2.get_len ();
    unsigned int record_len = len1 + len2;

    unsigned int klass1 = (this+classDef1).get_class (buffer->cur().codepoint);
    unsigned int klass2 = (this+classDef2).get_class (buffer->info[skippy_iter.idx].codepoint);
    if (unlikely (klass1 >= class1Count || klass2 >= class2Count)) return_trace (false);

    buffer->unsafe_to_break (buffer->idx, skippy_iter.idx + 1);
    const Value *v = &values[record_len * (klass1 * class2Count + klass2)];
    valueFormat1.apply_value (c, this, v, buffer->cur_pos());
    valueFormat2.apply_value (c, this, v + len1, buffer->pos[skippy_iter.idx]);

    buffer->idx = skippy_iter.idx;
    if (len2)
      buffer->idx++;

    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!(c->check_struct (this)
       && coverage.sanitize (c, this)
       && classDef1.sanitize (c, this)
       && classDef2.sanitize (c, this))) return_trace (false);

    unsigned int len1 = valueFormat1.get_len ();
    unsigned int len2 = valueFormat2.get_len ();
    unsigned int stride = len1 + len2;
    unsigned int record_size = valueFormat1.get_size () + valueFormat2.get_size ();
    unsigned int count = (unsigned int) class1Count * (unsigned int) class2Count;
    return_trace (c->check_array (values, record_size, count) &&
		  valueFormat1.sanitize_values_stride_unsafe (c, this, &values[0], count, stride) &&
		  valueFormat2.sanitize_values_stride_unsafe (c, this, &values[len1], count, stride));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 2 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of subtable */
  ValueFormat	valueFormat1;		/* ValueRecord definition--for the
					 * first glyph of the pair--may be zero
					 * (0) */
  ValueFormat	valueFormat2;		/* ValueRecord definition--for the
					 * second glyph of the pair--may be
					 * zero (0) */
  OffsetTo<ClassDef>
		classDef1;		/* Offset to ClassDef table--from
					 * beginning of PairPos subtable--for
					 * the first glyph of the pair */
  OffsetTo<ClassDef>
		classDef2;		/* Offset to ClassDef table--from
					 * beginning of PairPos subtable--for
					 * the second glyph of the pair */
  HBUINT16	class1Count;		/* Number of classes in ClassDef1
					 * table--includes Class0 */
  HBUINT16	class2Count;		/* Number of classes in ClassDef2
					 * table--includes Class0 */
  ValueRecord	values;			/* Matrix of value pairs:
					 * class1-major, class2-minor,
					 * Each entry has value1 and value2 */
  public:
  DEFINE_SIZE_ARRAY (16, values);
};

struct PairPos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    case 2: return_trace (c->dispatch (u.format2));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  PairPosFormat1	format1;
  PairPosFormat2	format2;
  } u;
};


struct EntryExitRecord
{
  friend struct CursivePosFormat1;

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (entryAnchor.sanitize (c, base) && exitAnchor.sanitize (c, base));
  }

  protected:
  OffsetTo<Anchor>
		entryAnchor;		/* Offset to EntryAnchor table--from
					 * beginning of CursivePos
					 * subtable--may be NULL */
  OffsetTo<Anchor>
		exitAnchor;		/* Offset to ExitAnchor table--from
					 * beginning of CursivePos
					 * subtable--may be NULL */
  public:
  DEFINE_SIZE_STATIC (4);
};

static void
reverse_cursive_minor_offset (hb_glyph_position_t *pos, unsigned int i, hb_direction_t direction, unsigned int new_parent);

struct CursivePosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+coverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;

    const EntryExitRecord &this_record = entryExitRecord[(this+coverage).get_coverage  (buffer->cur().codepoint)];
    if (!this_record.exitAnchor) return_trace (false);

    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    if (!skippy_iter.next ()) return_trace (false);

    const EntryExitRecord &next_record = entryExitRecord[(this+coverage).get_coverage  (buffer->info[skippy_iter.idx].codepoint)];
    if (!next_record.entryAnchor) return_trace (false);

    unsigned int i = buffer->idx;
    unsigned int j = skippy_iter.idx;

    buffer->unsafe_to_break (i, j);
    float entry_x, entry_y, exit_x, exit_y;
    (this+this_record.exitAnchor).get_anchor (c, buffer->info[i].codepoint, &exit_x, &exit_y);
    (this+next_record.entryAnchor).get_anchor (c, buffer->info[j].codepoint, &entry_x, &entry_y);

    hb_glyph_position_t *pos = buffer->pos;

    hb_position_t d;
    /* Main-direction adjustment */
    switch (c->direction) {
      case HB_DIRECTION_LTR:
	pos[i].x_advance  = round (exit_x) + pos[i].x_offset;

	d = round (entry_x) + pos[j].x_offset;
	pos[j].x_advance -= d;
	pos[j].x_offset  -= d;
	break;
      case HB_DIRECTION_RTL:
	d = round (exit_x) + pos[i].x_offset;
	pos[i].x_advance -= d;
	pos[i].x_offset  -= d;

	pos[j].x_advance  = round (entry_x) + pos[j].x_offset;
	break;
      case HB_DIRECTION_TTB:
	pos[i].y_advance  = round (exit_y) + pos[i].y_offset;

	d = round (entry_y) + pos[j].y_offset;
	pos[j].y_advance -= d;
	pos[j].y_offset  -= d;
	break;
      case HB_DIRECTION_BTT:
	d = round (exit_y) + pos[i].y_offset;
	pos[i].y_advance -= d;
	pos[i].y_offset  -= d;

	pos[j].y_advance  = round (entry_y);
	break;
      case HB_DIRECTION_INVALID:
      default:
	break;
    }

    /* Cross-direction adjustment */

    /* We attach child to parent (think graph theory and rooted trees whereas
     * the root stays on baseline and each node aligns itself against its
     * parent.
     *
     * Optimize things for the case of RightToLeft, as that's most common in
     * Arabinc. */
    unsigned int child  = i;
    unsigned int parent = j;
    hb_position_t x_offset = entry_x - exit_x;
    hb_position_t y_offset = entry_y - exit_y;
    if  (!(c->lookup_props & LookupFlag::RightToLeft))
    {
      unsigned int k = child;
      child = parent;
      parent = k;
      x_offset = -x_offset;
      y_offset = -y_offset;
    }

    /* If child was already connected to someone else, walk through its old
     * chain and reverse the link direction, such that the whole tree of its
     * previous connection now attaches to new parent.  Watch out for case
     * where new parent is on the path from old chain...
     */
    reverse_cursive_minor_offset (pos, child, c->direction, parent);

    pos[child].attach_type() = ATTACH_TYPE_CURSIVE;
    pos[child].attach_chain() = (int) parent - (int) child;
    buffer->scratch_flags |= HB_BUFFER_SCRATCH_FLAG_HAS_GPOS_ATTACHMENT;
    if (likely (HB_DIRECTION_IS_HORIZONTAL (c->direction)))
      pos[child].y_offset = y_offset;
    else
      pos[child].x_offset = x_offset;

    buffer->idx = j;
    return_trace (true);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (coverage.sanitize (c, this) && entryExitRecord.sanitize (c, this));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		coverage;		/* Offset to Coverage table--from
					 * beginning of subtable */
  ArrayOf<EntryExitRecord>
		entryExitRecord;	/* Array of EntryExit records--in
					 * Coverage Index order */
  public:
  DEFINE_SIZE_ARRAY (6, entryExitRecord);
};

struct CursivePos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  CursivePosFormat1	format1;
  } u;
};


typedef AnchorMatrix BaseArray;		/* base-major--
					 * in order of BaseCoverage Index--,
					 * mark-minor--
					 * ordered by class--zero-based. */

struct MarkBasePosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+markCoverage).add_coverage (c->input))) return;
    if (unlikely (!(this+baseCoverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+markCoverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark_index = (this+markCoverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark_index == NOT_COVERED)) return_trace (false);

    /* Now we search backwards for a non-mark glyph */
    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    skippy_iter.set_lookup_props (LookupFlag::IgnoreMarks);
    do {
      if (!skippy_iter.prev ()) return_trace (false);
      /* We only want to attach to the first of a MultipleSubst sequence.
       * https://github.com/harfbuzz/harfbuzz/issues/740
       * Reject others...
       * ...but stop if we find a mark in the MultipleSubst sequence:
       * https://github.com/harfbuzz/harfbuzz/issues/1020 */
      if (!_hb_glyph_info_multiplied (&buffer->info[skippy_iter.idx]) ||
	  0 == _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx]) ||
	  (skippy_iter.idx == 0 ||
	   _hb_glyph_info_is_mark (&buffer->info[skippy_iter.idx - 1]) ||
	   _hb_glyph_info_get_lig_id (&buffer->info[skippy_iter.idx]) !=
	   _hb_glyph_info_get_lig_id (&buffer->info[skippy_iter.idx - 1]) ||
	   _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx]) !=
	   _hb_glyph_info_get_lig_comp (&buffer->info[skippy_iter.idx - 1]) + 1
	   ))
	break;
      skippy_iter.reject ();
    } while (1);

    /* Checking that matched glyph is actually a base glyph by GDEF is too strong; disabled */
    //if (!_hb_glyph_info_is_base_glyph (&buffer->info[skippy_iter.idx])) { return_trace (false); }

    unsigned int base_index = (this+baseCoverage).get_coverage  (buffer->info[skippy_iter.idx].codepoint);
    if (base_index == NOT_COVERED) return_trace (false);

    return_trace ((this+markArray).apply (c, mark_index, base_index, this+baseArray, classCount, skippy_iter.idx));
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  markCoverage.sanitize (c, this) &&
		  baseCoverage.sanitize (c, this) &&
		  markArray.sanitize (c, this) &&
		  baseArray.sanitize (c, this, (unsigned int) classCount));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		markCoverage;		/* Offset to MarkCoverage table--from
					 * beginning of MarkBasePos subtable */
  OffsetTo<Coverage>
		baseCoverage;		/* Offset to BaseCoverage table--from
					 * beginning of MarkBasePos subtable */
  HBUINT16	classCount;		/* Number of classes defined for marks */
  OffsetTo<MarkArray>
		markArray;		/* Offset to MarkArray table--from
					 * beginning of MarkBasePos subtable */
  OffsetTo<BaseArray>
		baseArray;		/* Offset to BaseArray table--from
					 * beginning of MarkBasePos subtable */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct MarkBasePos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  MarkBasePosFormat1	format1;
  } u;
};


typedef AnchorMatrix LigatureAttach;	/* component-major--
					 * in order of writing direction--,
					 * mark-minor--
					 * ordered by class--zero-based. */

typedef OffsetListOf<LigatureAttach> LigatureArray;
					/* Array of LigatureAttach
					 * tables ordered by
					 * LigatureCoverage Index */

struct MarkLigPosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+markCoverage).add_coverage (c->input))) return;
    if (unlikely (!(this+ligatureCoverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+markCoverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark_index = (this+markCoverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark_index == NOT_COVERED)) return_trace (false);

    /* Now we search backwards for a non-mark glyph */
    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    skippy_iter.set_lookup_props (LookupFlag::IgnoreMarks);
    if (!skippy_iter.prev ()) return_trace (false);

    /* Checking that matched glyph is actually a ligature by GDEF is too strong; disabled */
    //if (!_hb_glyph_info_is_ligature (&buffer->info[skippy_iter.idx])) { return_trace (false); }

    unsigned int j = skippy_iter.idx;
    unsigned int lig_index = (this+ligatureCoverage).get_coverage  (buffer->info[j].codepoint);
    if (lig_index == NOT_COVERED) return_trace (false);

    const LigatureArray& lig_array = this+ligatureArray;
    const LigatureAttach& lig_attach = lig_array[lig_index];

    /* Find component to attach to */
    unsigned int comp_count = lig_attach.rows;
    if (unlikely (!comp_count)) return_trace (false);

    /* We must now check whether the ligature ID of the current mark glyph
     * is identical to the ligature ID of the found ligature.  If yes, we
     * can directly use the component index.  If not, we attach the mark
     * glyph to the last component of the ligature. */
    unsigned int comp_index;
    unsigned int lig_id = _hb_glyph_info_get_lig_id (&buffer->info[j]);
    unsigned int mark_id = _hb_glyph_info_get_lig_id (&buffer->cur());
    unsigned int mark_comp = _hb_glyph_info_get_lig_comp (&buffer->cur());
    if (lig_id && lig_id == mark_id && mark_comp > 0)
      comp_index = MIN (comp_count, _hb_glyph_info_get_lig_comp (&buffer->cur())) - 1;
    else
      comp_index = comp_count - 1;

    return_trace ((this+markArray).apply (c, mark_index, comp_index, lig_attach, classCount, j));
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  markCoverage.sanitize (c, this) &&
		  ligatureCoverage.sanitize (c, this) &&
		  markArray.sanitize (c, this) &&
		  ligatureArray.sanitize (c, this, (unsigned int) classCount));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		markCoverage;		/* Offset to Mark Coverage table--from
					 * beginning of MarkLigPos subtable */
  OffsetTo<Coverage>
		ligatureCoverage;	/* Offset to Ligature Coverage
					 * table--from beginning of MarkLigPos
					 * subtable */
  HBUINT16	classCount;		/* Number of defined mark classes */
  OffsetTo<MarkArray>
		markArray;		/* Offset to MarkArray table--from
					 * beginning of MarkLigPos subtable */
  OffsetTo<LigatureArray>
		ligatureArray;		/* Offset to LigatureArray table--from
					 * beginning of MarkLigPos subtable */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct MarkLigPos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  MarkLigPosFormat1	format1;
  } u;
};


typedef AnchorMatrix Mark2Array;	/* mark2-major--
					 * in order of Mark2Coverage Index--,
					 * mark1-minor--
					 * ordered by class--zero-based. */

struct MarkMarkPosFormat1
{
  inline void collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    if (unlikely (!(this+mark1Coverage).add_coverage (c->input))) return;
    if (unlikely (!(this+mark2Coverage).add_coverage (c->input))) return;
  }

  inline const Coverage &get_coverage (void) const
  {
    return this+mark1Coverage;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    hb_buffer_t *buffer = c->buffer;
    unsigned int mark1_index = (this+mark1Coverage).get_coverage  (buffer->cur().codepoint);
    if (likely (mark1_index == NOT_COVERED)) return_trace (false);

    /* now we search backwards for a suitable mark glyph until a non-mark glyph */
    hb_ot_apply_context_t::skipping_iterator_t &skippy_iter = c->iter_input;
    skippy_iter.reset (buffer->idx, 1);
    skippy_iter.set_lookup_props (c->lookup_props & ~LookupFlag::IgnoreFlags);
    if (!skippy_iter.prev ()) return_trace (false);

    if (!_hb_glyph_info_is_mark (&buffer->info[skippy_iter.idx])) { return_trace (false); }

    unsigned int j = skippy_iter.idx;

    unsigned int id1 = _hb_glyph_info_get_lig_id (&buffer->cur());
    unsigned int id2 = _hb_glyph_info_get_lig_id (&buffer->info[j]);
    unsigned int comp1 = _hb_glyph_info_get_lig_comp (&buffer->cur());
    unsigned int comp2 = _hb_glyph_info_get_lig_comp (&buffer->info[j]);

    if (likely (id1 == id2)) {
      if (id1 == 0) /* Marks belonging to the same base. */
	goto good;
      else if (comp1 == comp2) /* Marks belonging to the same ligature component. */
        goto good;
    } else {
      /* If ligature ids don't match, it may be the case that one of the marks
       * itself is a ligature.  In which case match. */
      if ((id1 > 0 && !comp1) || (id2 > 0 && !comp2))
	goto good;
    }

    /* Didn't match. */
    return_trace (false);

    good:
    unsigned int mark2_index = (this+mark2Coverage).get_coverage  (buffer->info[j].codepoint);
    if (mark2_index == NOT_COVERED) return_trace (false);

    return_trace ((this+mark1Array).apply (c, mark1_index, mark2_index, this+mark2Array, classCount, j));
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  mark1Coverage.sanitize (c, this) &&
		  mark2Coverage.sanitize (c, this) &&
		  mark1Array.sanitize (c, this) &&
		  mark2Array.sanitize (c, this, (unsigned int) classCount));
  }

  protected:
  HBUINT16	format;			/* Format identifier--format = 1 */
  OffsetTo<Coverage>
		mark1Coverage;		/* Offset to Combining Mark1 Coverage
					 * table--from beginning of MarkMarkPos
					 * subtable */
  OffsetTo<Coverage>
		mark2Coverage;		/* Offset to Combining Mark2 Coverage
					 * table--from beginning of MarkMarkPos
					 * subtable */
  HBUINT16	classCount;		/* Number of defined mark classes */
  OffsetTo<MarkArray>
		mark1Array;		/* Offset to Mark1Array table--from
					 * beginning of MarkMarkPos subtable */
  OffsetTo<Mark2Array>
		mark2Array;		/* Offset to Mark2Array table--from
					 * beginning of MarkMarkPos subtable */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct MarkMarkPos
{
  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  {
    TRACE_DISPATCH (this, u.format);
    if (unlikely (!c->may_dispatch (this, &u.format))) return_trace (c->no_dispatch_return_value ());
    switch (u.format) {
    case 1: return_trace (c->dispatch (u.format1));
    default:return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		format;		/* Format identifier */
  MarkMarkPosFormat1	format1;
  } u;
};


struct ContextPos : Context {};

struct ChainContextPos : ChainContext {};

struct ExtensionPos : Extension<ExtensionPos>
{
  typedef struct PosLookupSubTable LookupSubTable;
};



/*
 * PosLookup
 */


struct PosLookupSubTable
{
  friend struct PosLookup;

  enum Type {
    Single		= 1,
    Pair		= 2,
    Cursive		= 3,
    MarkBase		= 4,
    MarkLig		= 5,
    MarkMark		= 6,
    Context		= 7,
    ChainContext	= 8,
    Extension		= 9
  };

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c, unsigned int lookup_type) const
  {
    TRACE_DISPATCH (this, lookup_type);
    if (unlikely (!c->may_dispatch (this, &u.sub_format))) return_trace (c->no_dispatch_return_value ());
    switch (lookup_type) {
    case Single:		return_trace (u.single.dispatch (c));
    case Pair:			return_trace (u.pair.dispatch (c));
    case Cursive:		return_trace (u.cursive.dispatch (c));
    case MarkBase:		return_trace (u.markBase.dispatch (c));
    case MarkLig:		return_trace (u.markLig.dispatch (c));
    case MarkMark:		return_trace (u.markMark.dispatch (c));
    case Context:		return_trace (u.context.dispatch (c));
    case ChainContext:		return_trace (u.chainContext.dispatch (c));
    case Extension:		return_trace (u.extension.dispatch (c));
    default:			return_trace (c->default_return_value ());
    }
  }

  protected:
  union {
  HBUINT16		sub_format;
  SinglePos		single;
  PairPos		pair;
  CursivePos		cursive;
  MarkBasePos		markBase;
  MarkLigPos		markLig;
  MarkMarkPos		markMark;
  ContextPos		context;
  ChainContextPos	chainContext;
  ExtensionPos		extension;
  } u;
  public:
  DEFINE_SIZE_UNION (2, sub_format);
};


struct PosLookup : Lookup
{
  inline const PosLookupSubTable& get_subtable (unsigned int i) const
  { return Lookup::get_subtable<PosLookupSubTable> (i); }

  inline bool is_reverse (void) const
  {
    return false;
  }

  inline bool apply (hb_ot_apply_context_t *c) const
  {
    TRACE_APPLY (this);
    return_trace (dispatch (c));
  }

  inline hb_collect_glyphs_context_t::return_t collect_glyphs (hb_collect_glyphs_context_t *c) const
  {
    TRACE_COLLECT_GLYPHS (this);
    return_trace (dispatch (c));
  }

  template <typename set_t>
  inline void add_coverage (set_t *glyphs) const
  {
    hb_add_coverage_context_t<set_t> c (glyphs);
    dispatch (&c);
  }

  static bool apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index);

  template <typename context_t>
  static inline typename context_t::return_t dispatch_recurse_func (context_t *c, unsigned int lookup_index);

  template <typename context_t>
  inline typename context_t::return_t dispatch (context_t *c) const
  { return Lookup::dispatch<PosLookupSubTable> (c); }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!Lookup::sanitize (c))) return_trace (false);
    return_trace (dispatch (c));
  }
};

typedef OffsetListOf<PosLookup> PosLookupList;

/*
 * GPOS -- Glyph Positioning
 * https://docs.microsoft.com/en-us/typography/opentype/spec/gpos
 */

struct GPOS : GSUBGPOS
{
  static const hb_tag_t tableTag	= HB_OT_TAG_GPOS;

  inline const PosLookup& get_lookup (unsigned int i) const
  { return CastR<PosLookup> (GSUBGPOS::get_lookup (i)); }

  static inline void position_start (hb_font_t *font, hb_buffer_t *buffer);
  static inline void position_finish_advances (hb_font_t *font, hb_buffer_t *buffer);
  static inline void position_finish_offsets (hb_font_t *font, hb_buffer_t *buffer);

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!GSUBGPOS::sanitize (c))) return_trace (false);
    const OffsetTo<PosLookupList> &list = CastR<OffsetTo<PosLookupList> > (lookupList);
    return_trace (list.sanitize (c, this));
  }
};


static void
reverse_cursive_minor_offset (hb_glyph_position_t *pos, unsigned int i, hb_direction_t direction, unsigned int new_parent)
{
  int chain = pos[i].attach_chain(), type = pos[i].attach_type();
  if (likely (!chain || 0 == (type & ATTACH_TYPE_CURSIVE)))
    return;

  pos[i].attach_chain() = 0;

  unsigned int j = (int) i + chain;

  /* Stop if we see new parent in the chain. */
  if (j == new_parent)
    return;

  reverse_cursive_minor_offset (pos, j, direction, new_parent);

  if (HB_DIRECTION_IS_HORIZONTAL (direction))
    pos[j].y_offset = -pos[i].y_offset;
  else
    pos[j].x_offset = -pos[i].x_offset;

  pos[j].attach_chain() = -chain;
  pos[j].attach_type() = type;
}
static void
propagate_attachment_offsets (hb_glyph_position_t *pos, unsigned int i, hb_direction_t direction)
{
  /* Adjusts offsets of attached glyphs (both cursive and mark) to accumulate
   * offset of glyph they are attached to. */
  int chain = pos[i].attach_chain(), type = pos[i].attach_type();
  if (likely (!chain))
    return;

  unsigned int j = (int) i + chain;

  pos[i].attach_chain() = 0;

  propagate_attachment_offsets (pos, j, direction);

  assert (!!(type & ATTACH_TYPE_MARK) ^ !!(type & ATTACH_TYPE_CURSIVE));

  if (type & ATTACH_TYPE_CURSIVE)
  {
    if (HB_DIRECTION_IS_HORIZONTAL (direction))
      pos[i].y_offset += pos[j].y_offset;
    else
      pos[i].x_offset += pos[j].x_offset;
  }
  else /*if (type & ATTACH_TYPE_MARK)*/
  {
    pos[i].x_offset += pos[j].x_offset;
    pos[i].y_offset += pos[j].y_offset;

    assert (j < i);
    if (HB_DIRECTION_IS_FORWARD (direction))
      for (unsigned int k = j; k < i; k++) {
	pos[i].x_offset -= pos[k].x_advance;
	pos[i].y_offset -= pos[k].y_advance;
      }
    else
      for (unsigned int k = j + 1; k < i + 1; k++) {
	pos[i].x_offset += pos[k].x_advance;
	pos[i].y_offset += pos[k].y_advance;
      }
  }
}

void
GPOS::position_start (hb_font_t *font HB_UNUSED, hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
  for (unsigned int i = 0; i < count; i++)
    buffer->pos[i].attach_chain() = buffer->pos[i].attach_type() = 0;
}

void
GPOS::position_finish_advances (hb_font_t *font HB_UNUSED, hb_buffer_t *buffer)
{
  //_hb_buffer_assert_gsubgpos_vars (buffer);
}

void
GPOS::position_finish_offsets (hb_font_t *font HB_UNUSED, hb_buffer_t *buffer)
{
  _hb_buffer_assert_gsubgpos_vars (buffer);

  unsigned int len;
  hb_glyph_position_t *pos = hb_buffer_get_glyph_positions (buffer, &len);
  hb_direction_t direction = buffer->props.direction;

  /* Handle attachments */
  if (buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_GPOS_ATTACHMENT)
    for (unsigned int i = 0; i < len; i++)
      propagate_attachment_offsets (pos, i, direction);
}


/* Out-of-class implementation for methods recursing */

template <typename context_t>
/*static*/ inline typename context_t::return_t PosLookup::dispatch_recurse_func (context_t *c, unsigned int lookup_index)
{
  const GPOS &gpos = *(hb_ot_layout_from_face (c->face)->table.GPOS);
  const PosLookup &l = gpos.get_lookup (lookup_index);
  return l.dispatch (c);
}

/*static*/ inline bool PosLookup::apply_recurse_func (hb_ot_apply_context_t *c, unsigned int lookup_index)
{
  const GPOS &gpos = *(hb_ot_layout_from_face (c->face)->table.GPOS);
  const PosLookup &l = gpos.get_lookup (lookup_index);
  unsigned int saved_lookup_props = c->lookup_props;
  unsigned int saved_lookup_index = c->lookup_index;
  c->set_lookup_index (lookup_index);
  c->set_lookup_props (l.get_props ());
  bool ret = l.dispatch (c);
  c->set_lookup_index (saved_lookup_index);
  c->set_lookup_props (saved_lookup_props);
  return ret;
}


#undef attach_chain
#undef attach_type


} /* namespace OT */


#endif /* HB_OT_LAYOUT_GPOS_TABLE_HH */
