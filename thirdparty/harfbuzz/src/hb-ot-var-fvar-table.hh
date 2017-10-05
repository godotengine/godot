/*
 * Copyright © 2017  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_OT_VAR_FVAR_TABLE_HH
#define HB_OT_VAR_FVAR_TABLE_HH

#include "hb-open-type-private.hh"

/*
 * fvar -- Font Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/fvar
 */

#define HB_OT_TAG_fvar HB_TAG('f','v','a','r')


namespace OT {


struct InstanceRecord
{
  inline bool sanitize (hb_sanitize_context_t *c, unsigned int axis_count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  c->check_array (coordinates, coordinates[0].static_size, axis_count));
  }

  protected:
  NameID	subfamilyNameID;/* The name ID for entries in the 'name' table
				 * that provide subfamily names for this instance. */
  HBUINT16	reserved;	/* Reserved for future use — set to 0. */
  Fixed		coordinates[VAR];/* The coordinates array for this instance. */
  //NameID	postScriptNameIDX;/*Optional. The name ID for entries in the 'name'
  //				  * table that provide PostScript names for this
  //				  * instance. */

  public:
  DEFINE_SIZE_ARRAY (4, coordinates);
};

struct AxisRecord
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  public:
  Tag		axisTag;	/* Tag identifying the design variation for the axis. */
  Fixed		minValue;	/* The minimum coordinate value for the axis. */
  Fixed		defaultValue;	/* The default coordinate value for the axis. */
  Fixed		maxValue;	/* The maximum coordinate value for the axis. */
  HBUINT16	reserved;	/* Reserved for future use — set to 0. */
  NameID	axisNameID;	/* The name ID for entries in the 'name' table that
				 * provide a display name for this axis. */

  public:
  DEFINE_SIZE_STATIC (20);
};

struct fvar
{
  static const hb_tag_t tableTag	= HB_OT_TAG_fvar;

  inline bool has_data (void) const { return version.to_int () != 0; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  c->check_struct (this) &&
		  instanceSize >= axisCount * 4 + 4 &&
		  axisSize <= 1024 && /* Arbitrary, just to simplify overflow checks. */
		  instanceSize <= 1024 && /* Arbitrary, just to simplify overflow checks. */
		  c->check_range (this, things) &&
		  c->check_range (&StructAtOffset<char> (this, things),
				  axisCount * axisSize + instanceCount * instanceSize));
  }

  inline unsigned int get_axis_count (void) const
  { return axisCount; }

  inline bool get_axis (unsigned int index, hb_ot_var_axis_t *info) const
  {
    if (unlikely (index >= axisCount))
      return false;

    if (info)
    {
      const AxisRecord &axis = get_axes ()[index];
      info->tag = axis.axisTag;
      info->name_id =  axis.axisNameID;
      info->default_value = axis.defaultValue / 65536.;
      /* Ensure order, to simplify client math. */
      info->min_value = MIN<float> (info->default_value, axis.minValue / 65536.);
      info->max_value = MAX<float> (info->default_value, axis.maxValue / 65536.);
    }

    return true;
  }

  inline unsigned int get_axis_infos (unsigned int      start_offset,
				      unsigned int     *axes_count /* IN/OUT */,
				      hb_ot_var_axis_t *axes_array /* OUT */) const
  {
    if (axes_count)
    {
      unsigned int count = axisCount;
      start_offset = MIN (start_offset, count);

      count -= start_offset;
      axes_array += start_offset;

      count = MIN (count, *axes_count);
      *axes_count = count;

      for (unsigned int i = 0; i < count; i++)
	get_axis (start_offset + i, axes_array + i);
    }
    return axisCount;
  }

  inline bool find_axis (hb_tag_t tag, unsigned int *index, hb_ot_var_axis_t *info) const
  {
    const AxisRecord *axes = get_axes ();
    unsigned int count = get_axis_count ();
    for (unsigned int i = 0; i < count; i++)
      if (axes[i].axisTag == tag)
      {
        if (index)
	  *index = i;
	return get_axis (i, info);
      }
    if (index)
      *index = HB_OT_VAR_NO_AXIS_INDEX;
    return false;
  }

  inline int normalize_axis_value (unsigned int axis_index, float v) const
  {
    hb_ot_var_axis_t axis;
    if (!get_axis (axis_index, &axis))
      return 0;

    v = MAX (MIN (v, axis.max_value), axis.min_value); /* Clamp. */

    if (v == axis.default_value)
      return 0;
    else if (v < axis.default_value)
      v = (v - axis.default_value) / (axis.default_value - axis.min_value);
    else
      v = (v - axis.default_value) / (axis.max_value - axis.default_value);
    return (int) (v * 16384. + (v >= 0. ? .5 : -.5));
  }

  protected:
  inline const AxisRecord * get_axes (void) const
  { return &StructAtOffset<AxisRecord> (this, things); }

  inline const InstanceRecord * get_instances (void) const
  { return &StructAtOffset<InstanceRecord> (get_axes () + axisCount, 0); }

  protected:
  FixedVersion<>version;	/* Version of the fvar table
				 * initially set to 0x00010000u */
  Offset16	things;		/* Offset in bytes from the beginning of the table
				 * to the start of the AxisRecord array. */
  HBUINT16	reserved;	/* This field is permanently reserved. Set to 2. */
  HBUINT16	axisCount;	/* The number of variation axes in the font (the
				 * number of records in the axes array). */
  HBUINT16	axisSize;	/* The size in bytes of each VariationAxisRecord —
				 * set to 20 (0x0014) for this version. */
  HBUINT16	instanceCount;	/* The number of named instances defined in the font
				 * (the number of records in the instances array). */
  HBUINT16	instanceSize;	/* The size in bytes of each InstanceRecord — set
				 * to either axisCount * sizeof(Fixed) + 4, or to
				 * axisCount * sizeof(Fixed) + 6. */

  public:
  DEFINE_SIZE_STATIC (16);
};

} /* namespace OT */


#endif /* HB_OT_VAR_FVAR_TABLE_HH */
