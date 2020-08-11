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

#include "hb-open-type.hh"

/*
 * fvar -- Font Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/fvar
 */

#define HB_OT_TAG_fvar HB_TAG('f','v','a','r')


namespace OT {


struct InstanceRecord
{
  friend struct fvar;

  hb_array_t<const HBFixed> get_coordinates (unsigned int axis_count) const
  { return coordinatesZ.as_array (axis_count); }

  bool sanitize (hb_sanitize_context_t *c, unsigned int axis_count) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  c->check_array (coordinatesZ.arrayZ, axis_count));
  }

  protected:
  NameID	subfamilyNameID;/* The name ID for entries in the 'name' table
				 * that provide subfamily names for this instance. */
  HBUINT16	flags;		/* Reserved for future use — set to 0. */
  UnsizedArrayOf<HBFixed>
		coordinatesZ;	/* The coordinates array for this instance. */
  //NameID	postScriptNameIDX;/*Optional. The name ID for entries in the 'name'
  //				  * table that provide PostScript names for this
  //				  * instance. */

  public:
  DEFINE_SIZE_UNBOUNDED (4);
};

struct AxisRecord
{
  int cmp (hb_tag_t key) const { return axisTag.cmp (key); }

  enum
  {
    AXIS_FLAG_HIDDEN	= 0x0001,
  };

#ifndef HB_DISABLE_DEPRECATED
  void get_axis_deprecated (hb_ot_var_axis_t *info) const
  {
    info->tag = axisTag;
    info->name_id = axisNameID;
    get_coordinates (info->min_value, info->default_value, info->max_value);
  }
#endif

  void get_axis_info (unsigned axis_index, hb_ot_var_axis_info_t *info) const
  {
    info->axis_index = axis_index;
    info->tag = axisTag;
    info->name_id = axisNameID;
    info->flags = (hb_ot_var_axis_flags_t) (unsigned int) flags;
    get_coordinates (info->min_value, info->default_value, info->max_value);
    info->reserved = 0;
  }

  int normalize_axis_value (float v) const
  {
    float min_value, default_value, max_value;
    get_coordinates (min_value, default_value, max_value);

    v = hb_clamp (v, min_value, max_value);

    if (v == default_value)
      return 0;
    else if (v < default_value)
      v = (v - default_value) / (default_value - min_value);
    else
      v = (v - default_value) / (max_value - default_value);
    return roundf (v * 16384.f);
  }

  float unnormalize_axis_value (int v) const
  {
    float min_value, default_value, max_value;
    get_coordinates (min_value, default_value, max_value);

    if (v == 0)
      return default_value;
    else if (v < 0)
      return v * (default_value - min_value) / 16384.f + default_value;
    else
      return v * (max_value - default_value) / 16384.f + default_value;
  }

  hb_ot_name_id_t get_name_id () const { return axisNameID; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  void get_coordinates (float &min, float &default_, float &max) const
  {
    default_ = defaultValue / 65536.f;
    /* Ensure order, to simplify client math. */
    min = hb_min (default_, minValue / 65536.f);
    max = hb_max (default_, maxValue / 65536.f);
  }

  protected:
  Tag		axisTag;	/* Tag identifying the design variation for the axis. */
  HBFixed	minValue;	/* The minimum coordinate value for the axis. */
  HBFixed	defaultValue;	/* The default coordinate value for the axis. */
  HBFixed	maxValue;	/* The maximum coordinate value for the axis. */
  HBUINT16	flags;		/* Axis flags. */
  NameID	axisNameID;	/* The name ID for entries in the 'name' table that
				 * provide a display name for this axis. */

  public:
  DEFINE_SIZE_STATIC (20);
};

struct fvar
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_fvar;

  bool has_data () const { return version.to_int (); }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  c->check_struct (this) &&
		  axisSize == 20 && /* Assumed in our code. */
		  instanceSize >= axisCount * 4 + 4 &&
		  get_axes ().sanitize (c) &&
		  c->check_range (get_instance (0), instanceCount, instanceSize));
  }

  unsigned int get_axis_count () const { return axisCount; }

#ifndef HB_DISABLE_DEPRECATED
  unsigned int get_axes_deprecated (unsigned int      start_offset,
				    unsigned int     *axes_count /* IN/OUT */,
				    hb_ot_var_axis_t *axes_array /* OUT */) const
  {
    if (axes_count)
    {
      hb_array_t<const AxisRecord> arr = get_axes ().sub_array (start_offset, axes_count);
      for (unsigned i = 0; i < arr.length; ++i)
	arr[i].get_axis_deprecated (&axes_array[i]);
    }
    return axisCount;
  }
#endif

  unsigned int get_axis_infos (unsigned int           start_offset,
			       unsigned int          *axes_count /* IN/OUT */,
			       hb_ot_var_axis_info_t *axes_array /* OUT */) const
  {
    if (axes_count)
    {
      hb_array_t<const AxisRecord> arr = get_axes ().sub_array (start_offset, axes_count);
      for (unsigned i = 0; i < arr.length; ++i)
	arr[i].get_axis_info (start_offset + i, &axes_array[i]);
    }
    return axisCount;
  }

#ifndef HB_DISABLE_DEPRECATED
  bool
  find_axis_deprecated (hb_tag_t tag, unsigned *axis_index, hb_ot_var_axis_t *info) const
  {
    unsigned i;
    if (!axis_index) axis_index = &i;
    *axis_index = HB_OT_VAR_NO_AXIS_INDEX;
    auto axes = get_axes ();
    return axes.lfind (tag, axis_index) && (axes[*axis_index].get_axis_deprecated (info), true);
  }
#endif

  bool
  find_axis_info (hb_tag_t tag, hb_ot_var_axis_info_t *info) const
  {
    unsigned i;
    auto axes = get_axes ();
    return axes.lfind (tag, &i) && (axes[i].get_axis_info (i, info), true);
  }

  int normalize_axis_value (unsigned int axis_index, float v) const
  { return get_axes ()[axis_index].normalize_axis_value (v); }

  float unnormalize_axis_value (unsigned int axis_index, int v) const
  { return get_axes ()[axis_index].unnormalize_axis_value (v); }

  unsigned int get_instance_count () const { return instanceCount; }

  hb_ot_name_id_t get_instance_subfamily_name_id (unsigned int instance_index) const
  {
    const InstanceRecord *instance = get_instance (instance_index);
    if (unlikely (!instance)) return HB_OT_NAME_ID_INVALID;
    return instance->subfamilyNameID;
  }

  hb_ot_name_id_t get_instance_postscript_name_id (unsigned int instance_index) const
  {
    const InstanceRecord *instance = get_instance (instance_index);
    if (unlikely (!instance)) return HB_OT_NAME_ID_INVALID;
    if (instanceSize >= axisCount * 4 + 6)
      return StructAfter<NameID> (instance->get_coordinates (axisCount));
    return HB_OT_NAME_ID_INVALID;
  }

  unsigned int get_instance_coords (unsigned int  instance_index,
				    unsigned int *coords_length, /* IN/OUT */
				    float        *coords         /* OUT */) const
  {
    const InstanceRecord *instance = get_instance (instance_index);
    if (unlikely (!instance))
    {
      if (coords_length)
	*coords_length = 0;
      return 0;
    }

    if (coords_length && *coords_length)
    {
      hb_array_t<const HBFixed> instanceCoords = instance->get_coordinates (axisCount)
							 .sub_array (0, *coords_length);
      for (unsigned int i = 0; i < instanceCoords.length; i++)
	coords[i] = instanceCoords.arrayZ[i].to_float ();
    }
    return axisCount;
  }

  void collect_name_ids (hb_set_t *nameids) const
  {
    if (!has_data ()) return;

    + get_axes ()
    | hb_map (&AxisRecord::get_name_id)
    | hb_sink (nameids)
    ;

    + hb_range ((unsigned) instanceCount)
    | hb_map ([this] (const unsigned _) { return get_instance_subfamily_name_id (_); })
    | hb_sink (nameids)
    ;

    + hb_range ((unsigned) instanceCount)
    | hb_map ([this] (const unsigned _) { return get_instance_postscript_name_id (_); })
    | hb_sink (nameids)
    ;
  }

  protected:
  hb_array_t<const AxisRecord> get_axes () const
  { return hb_array (&(this+firstAxis), axisCount); }

  const InstanceRecord *get_instance (unsigned int i) const
  {
    if (unlikely (i >= instanceCount)) return nullptr;
   return &StructAtOffset<InstanceRecord> (&StructAfter<InstanceRecord> (get_axes ()),
					   i * instanceSize);
  }

  protected:
  FixedVersion<>version;	/* Version of the fvar table
				 * initially set to 0x00010000u */
  OffsetTo<AxisRecord>
		firstAxis;	/* Offset in bytes from the beginning of the table
				 * to the start of the AxisRecord array. */
  HBUINT16	reserved;	/* This field is permanently reserved. Set to 2. */
  HBUINT16	axisCount;	/* The number of variation axes in the font (the
				 * number of records in the axes array). */
  HBUINT16	axisSize;	/* The size in bytes of each VariationAxisRecord —
				 * set to 20 (0x0014) for this version. */
  HBUINT16	instanceCount;	/* The number of named instances defined in the font
				 * (the number of records in the instances array). */
  HBUINT16	instanceSize;	/* The size in bytes of each InstanceRecord — set
				 * to either axisCount * sizeof(HBFixed) + 4, or to
				 * axisCount * sizeof(HBFixed) + 6. */

  public:
  DEFINE_SIZE_STATIC (16);
};

} /* namespace OT */


#endif /* HB_OT_VAR_FVAR_TABLE_HH */
