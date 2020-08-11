/*
 * Copyright © 2018  Ebrahim Byagowi
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
 */

#ifndef HB_OT_STAT_TABLE_HH
#define HB_OT_STAT_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-common.hh"

/*
 * STAT -- Style Attributes
 * https://docs.microsoft.com/en-us/typography/opentype/spec/stat
 */
#define HB_OT_TAG_STAT HB_TAG('S','T','A','T')


namespace OT {

enum
{
  OLDER_SIBLING_FONT_ATTRIBUTE = 0x0001,	/* If set, this axis value table
						 * provides axis value information
						 * that is applicable to other fonts
						 * within the same font family. This
						 * is used if the other fonts were
						 * released earlier and did not include
						 * information about values for some axis.
						 * If newer versions of the other
						 * fonts include the information
						 * themselves and are present,
						 * then this record is ignored. */
  ELIDABLE_AXIS_VALUE_NAME = 0x0002		/* If set, it indicates that the axis
						 * value represents the “normal” value
						 * for the axis and may be omitted when
						 * composing name strings. */
  // Reserved = 0xFFFC				/* Reserved for future use — set to zero. */
};

struct AxisValueFormat1
{
  unsigned int get_axis_index () const { return axisIndex; }
  float get_value ()             const { return value.to_float (); }

  hb_ot_name_id_t get_value_name_id () const { return valueNameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier — set to 1. */
  HBUINT16	axisIndex;	/* Zero-base index into the axis record array
				 * identifying the axis of design variation
				 * to which the axis value record applies.
				 * Must be less than designAxisCount. */
  HBUINT16	flags;		/* Flags — see below for details. */
  NameID	valueNameID;	/* The name ID for entries in the 'name' table
				 * that provide a display string for this
				 * attribute value. */
  HBFixed	value;		/* A numeric value for this attribute value. */
  public:
  DEFINE_SIZE_STATIC (12);
};

struct AxisValueFormat2
{
  unsigned int get_axis_index () const { return axisIndex; }
  float get_value ()             const { return nominalValue.to_float (); }

  hb_ot_name_id_t get_value_name_id () const { return valueNameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier — set to 2. */
  HBUINT16	axisIndex;	/* Zero-base index into the axis record array
				 * identifying the axis of design variation
				 * to which the axis value record applies.
				 * Must be less than designAxisCount. */
  HBUINT16	flags;		/* Flags — see below for details. */
  NameID	valueNameID;	/* The name ID for entries in the 'name' table
				 * that provide a display string for this
				 * attribute value. */
  HBFixed	nominalValue;	/* A numeric value for this attribute value. */
  HBFixed	rangeMinValue;	/* The minimum value for a range associated
				 * with the specified name ID. */
  HBFixed	rangeMaxValue;	/* The maximum value for a range associated
				 * with the specified name ID. */
  public:
  DEFINE_SIZE_STATIC (20);
};

struct AxisValueFormat3
{
  unsigned int get_axis_index () const { return axisIndex; }
  float get_value ()             const { return value.to_float (); }

  hb_ot_name_id_t get_value_name_id () const { return valueNameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier — set to 3. */
  HBUINT16	axisIndex;	/* Zero-base index into the axis record array
				 * identifying the axis of design variation
				 * to which the axis value record applies.
				 * Must be less than designAxisCount. */
  HBUINT16	flags;		/* Flags — see below for details. */
  NameID	valueNameID;	/* The name ID for entries in the 'name' table
				 * that provide a display string for this
				 * attribute value. */
  HBFixed	value;		/* A numeric value for this attribute value. */
  HBFixed	linkedValue;	/* The numeric value for a style-linked mapping
				 * from this value. */
  public:
  DEFINE_SIZE_STATIC (16);
};

struct AxisValueRecord
{
  unsigned int get_axis_index () const { return axisIndex; }
  float get_value ()             const { return value.to_float (); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	axisIndex;	/* Zero-base index into the axis record array
				 * identifying the axis to which this value
				 * applies. Must be less than designAxisCount. */
  HBFixed	value;		/* A numeric value for this attribute value. */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct AxisValueFormat4
{
  const AxisValueRecord &get_axis_record (unsigned int axis_index) const
  { return axisValues.as_array (axisCount)[axis_index]; }

  hb_ot_name_id_t get_value_name_id () const { return valueNameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier — set to 4. */
  HBUINT16	axisCount;	/* The total number of axes contributing to
				 * this axis-values combination. */
  HBUINT16	flags;		/* Flags — see below for details. */
  NameID	valueNameID;	/* The name ID for entries in the 'name' table
				 * that provide a display string for this
				 * attribute value. */
  UnsizedArrayOf<AxisValueRecord>
		axisValues;	/* Array of AxisValue records that provide the
				 * combination of axis values, one for each
				 * contributing axis. */
  public:
  DEFINE_SIZE_ARRAY (8, axisValues);
};

struct AxisValue
{
  bool get_value (unsigned int axis_index) const
  {
    switch (u.format)
    {
    case 1: return u.format1.get_value ();
    case 2: return u.format2.get_value ();
    case 3: return u.format3.get_value ();
    case 4: return u.format4.get_axis_record (axis_index).get_value ();
    default:return 0;
    }
  }

  unsigned int get_axis_index () const
  {
    switch (u.format)
    {
    case 1: return u.format1.get_axis_index ();
    case 2: return u.format2.get_axis_index ();
    case 3: return u.format3.get_axis_index ();
    /* case 4: Makes more sense for variable fonts which are handled by fvar in hb-style */
    default:return -1;
    }
  }

  hb_ot_name_id_t get_value_name_id () const
  {
    switch (u.format)
    {
    case 1: return u.format1.get_value_name_id ();
    case 2: return u.format2.get_value_name_id ();
    case 3: return u.format3.get_value_name_id ();
    case 4: return u.format4.get_value_name_id ();
    default:return HB_OT_NAME_ID_INVALID;
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    switch (u.format)
    {
    case 1: return_trace (u.format1.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
    case 3: return_trace (u.format3.sanitize (c));
    case 4: return_trace (u.format4.sanitize (c));
    default:return_trace (true);
    }
  }

  protected:
  union
  {
  HBUINT16		format;
  AxisValueFormat1	format1;
  AxisValueFormat2	format2;
  AxisValueFormat3	format3;
  AxisValueFormat4	format4;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};

struct StatAxisRecord
{
  int cmp (hb_tag_t key) const { return tag.cmp (key); }

  hb_ot_name_id_t get_name_id () const { return nameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  Tag		tag;		/* A tag identifying the axis of design variation. */
  NameID	nameID;		/* The name ID for entries in the 'name' table that
				 * provide a display string for this axis. */
  HBUINT16	ordering;	/* A value that applications can use to determine
				 * primary sorting of face names, or for ordering
				 * of descriptors when composing family or face names. */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct STAT
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_STAT;

  bool has_data () const { return version.to_int (); }

  bool get_value (hb_tag_t tag, float *value) const
  {
    unsigned int axis_index;
    if (!get_design_axes ().lfind (tag, &axis_index)) return false;

    hb_array_t<const OffsetTo<AxisValue>> axis_values = get_axis_value_offsets ();
    for (unsigned int i = 0; i < axis_values.length; i++)
    {
      const AxisValue& axis_value = this+axis_values[i];
      if (axis_value.get_axis_index () == axis_index)
      {
	if (value)
	  *value = axis_value.get_value (axis_index);
	return true;
      }
    }
    return false;
  }

  unsigned get_design_axis_count () const { return designAxisCount; }

  hb_ot_name_id_t get_axis_record_name_id (unsigned axis_record_index) const
  {
    if (unlikely (axis_record_index >= designAxisCount)) return HB_OT_NAME_ID_INVALID;
    const StatAxisRecord &axis_record = get_design_axes ()[axis_record_index];
    return axis_record.get_name_id ();
  }

  unsigned get_axis_value_count () const { return axisValueCount; }

  hb_ot_name_id_t get_axis_value_name_id (unsigned axis_value_index) const
  {
    if (unlikely (axis_value_index >= axisValueCount)) return HB_OT_NAME_ID_INVALID;
    const AxisValue &axis_value = (this + get_axis_value_offsets ()[axis_value_index]);
    return axis_value.get_value_name_id ();
  }

  void collect_name_ids (hb_set_t *nameids_to_retain) const
  {
    if (!has_data ()) return;

    + get_design_axes ()
    | hb_map (&StatAxisRecord::get_name_id)
    | hb_sink (nameids_to_retain)
    ;

    + get_axis_value_offsets ()
    | hb_map (hb_add (&(this + offsetToAxisValueOffsets)))
    | hb_map (&AxisValue::get_value_name_id)
    | hb_sink (nameids_to_retain)
    ;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  version.major == 1 &&
			  version.minor > 0 &&
			  designAxesOffset.sanitize (c, this, designAxisCount) &&
			  offsetToAxisValueOffsets.sanitize (c, this, axisValueCount, &(this+offsetToAxisValueOffsets))));
  }

  protected:
  hb_array_t<const StatAxisRecord> const get_design_axes () const
  { return (this+designAxesOffset).as_array (designAxisCount); }

  hb_array_t<const OffsetTo<AxisValue>> const get_axis_value_offsets () const
  { return (this+offsetToAxisValueOffsets).as_array (axisValueCount); }


  protected:
  FixedVersion<>version;	/* Version of the stat table
				 * initially set to 0x00010002u */
  HBUINT16	designAxisSize;	/* The size in bytes of each axis record. */
  HBUINT16	designAxisCount;/* The number of design axis records. In a
				 * font with an 'fvar' table, this value must be
				 * greater than or equal to the axisCount value
				 * in the 'fvar' table. In all fonts, must
				 * be greater than zero if axisValueCount
				 * is greater than zero. */
  LNNOffsetTo<UnsizedArrayOf<StatAxisRecord>>
		designAxesOffset;
				/* Offset in bytes from the beginning of
				 * the STAT table to the start of the design
				 * axes array. If designAxisCount is zero,
				 * set to zero; if designAxisCount is greater
				 * than zero, must be greater than zero. */
  HBUINT16	axisValueCount;	/* The number of axis value tables. */
  LNNOffsetTo<UnsizedArrayOf<OffsetTo<AxisValue>>>
		offsetToAxisValueOffsets;
				/* Offset in bytes from the beginning of
				 * the STAT table to the start of the design
				 * axes value offsets array. If axisValueCount
				 * is zero, set to zero; if axisValueCount is
				 * greater than zero, must be greater than zero. */
  NameID	elidedFallbackNameID;
				/* Name ID used as fallback when projection of
				 * names into a particular font model produces
				 * a subfamily name containing only elidable
				 * elements. */
  public:
  DEFINE_SIZE_STATIC (20);
};


} /* namespace OT */


#endif /* HB_OT_STAT_TABLE_HH */
