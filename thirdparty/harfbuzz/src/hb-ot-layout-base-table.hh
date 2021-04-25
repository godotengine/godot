/*
 * Copyright © 2016  Elie Roux <elie.roux@telecom-bretagne.eu>
 * Copyright © 2018  Google, Inc.
 * Copyright © 2018-2019  Ebrahim Byagowi
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

#ifndef HB_OT_LAYOUT_BASE_TABLE_HH
#define HB_OT_LAYOUT_BASE_TABLE_HH

#include "hb-open-type.hh"
#include "hb-ot-layout-common.hh"

namespace OT {

/*
 * BASE -- Baseline
 * https://docs.microsoft.com/en-us/typography/opentype/spec/base
 */

struct BaseCoordFormat1
{
  hb_position_t get_coord () const { return coordinate; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 1 */
  FWORD		coordinate;	/* X or Y value, in design units */
  public:
  DEFINE_SIZE_STATIC (4);
};

struct BaseCoordFormat2
{
  hb_position_t get_coord () const
  {
    /* TODO */
    return coordinate;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 2 */
  FWORD		coordinate;	/* X or Y value, in design units */
  HBGlyphID	referenceGlyph;	/* Glyph ID of control glyph */
  HBUINT16	coordPoint;	/* Index of contour point on the
				 * reference glyph */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct BaseCoordFormat3
{
  hb_position_t get_coord (hb_font_t *font,
			   const VariationStore &var_store,
			   hb_direction_t direction) const
  {
    const Device &device = this+deviceTable;
    return coordinate + (HB_DIRECTION_IS_VERTICAL (direction) ?
			 device.get_y_delta (font, var_store) :
			 device.get_x_delta (font, var_store));
  }


  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  deviceTable.sanitize (c, this)));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 3 */
  FWORD		coordinate;	/* X or Y value, in design units */
  OffsetTo<Device>
		deviceTable;	/* Offset to Device table for X or
				 * Y value, from beginning of
				 * BaseCoord table (may be NULL). */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseCoord
{
  bool has_data () const { return u.format; }

  hb_position_t get_coord (hb_font_t            *font,
			   const VariationStore &var_store,
			   hb_direction_t        direction) const
  {
    switch (u.format) {
    case 1: return u.format1.get_coord ();
    case 2: return u.format2.get_coord ();
    case 3: return u.format3.get_coord (font, var_store, direction);
    default:return 0;
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!u.format.sanitize (c))) return_trace (false);
    switch (u.format) {
    case 1: return_trace (u.format1.sanitize (c));
    case 2: return_trace (u.format2.sanitize (c));
    case 3: return_trace (u.format3.sanitize (c));
    default:return_trace (false);
    }
  }

  protected:
  union {
  HBUINT16		format;
  BaseCoordFormat1	format1;
  BaseCoordFormat2	format2;
  BaseCoordFormat3	format3;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};

struct FeatMinMaxRecord
{
  int cmp (hb_tag_t key) const { return tag.cmp (key); }

  bool has_data () const { return tag; }

  void get_min_max (const BaseCoord **min, const BaseCoord **max) const
  {
    if (likely (min)) *min = &(this+minCoord);
    if (likely (max)) *max = &(this+maxCoord);
  }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  minCoord.sanitize (c, this) &&
			  maxCoord.sanitize (c, this)));
  }

  protected:
  Tag		tag;		/* 4-byte feature identification tag--must
				 * match feature tag in FeatureList */
  OffsetTo<BaseCoord>
		minCoord;	/* Offset to BaseCoord table that defines
				 * the minimum extent value, from beginning
				 * of MinMax table (may be NULL) */
  OffsetTo<BaseCoord>
		maxCoord;	/* Offset to BaseCoord table that defines
				 * the maximum extent value, from beginning
				 * of MinMax table (may be NULL) */
  public:
  DEFINE_SIZE_STATIC (8);

};

struct MinMax
{
  void get_min_max (hb_tag_t          feature_tag,
		    const BaseCoord **min,
		    const BaseCoord **max) const
  {
    const FeatMinMaxRecord &minMaxCoord = featMinMaxRecords.bsearch (feature_tag);
    if (minMaxCoord.has_data ())
      minMaxCoord.get_min_max (min, max);
    else
    {
      if (likely (min)) *min = &(this+minCoord);
      if (likely (max)) *max = &(this+maxCoord);
    }
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  minCoord.sanitize (c, this) &&
			  maxCoord.sanitize (c, this) &&
			  featMinMaxRecords.sanitize (c, this)));
  }

  protected:
  OffsetTo<BaseCoord>
		minCoord;	/* Offset to BaseCoord table that defines
				 * minimum extent value, from the beginning
				 * of MinMax table (may be NULL) */
  OffsetTo<BaseCoord>
		maxCoord;	/* Offset to BaseCoord table that defines
				 * maximum extent value, from the beginning
				 * of MinMax table (may be NULL) */
  SortedArrayOf<FeatMinMaxRecord>
		featMinMaxRecords;
				/* Array of FeatMinMaxRecords, in alphabetical
				 * order by featureTableTag */
  public:
  DEFINE_SIZE_ARRAY (6, featMinMaxRecords);
};

struct BaseValues
{
  const BaseCoord &get_base_coord (int baseline_tag_index) const
  {
    if (baseline_tag_index == -1) baseline_tag_index = defaultIndex;
    return this+baseCoords[baseline_tag_index];
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  baseCoords.sanitize (c, this)));
  }

  protected:
  Index		defaultIndex;	/* Index number of default baseline for this
				 * script — equals index position of baseline tag
				 * in baselineTags array of the BaseTagList */
  OffsetArrayOf<BaseCoord>
		baseCoords;	/* Number of BaseCoord tables defined — should equal
				 * baseTagCount in the BaseTagList
				 *
				 * Array of offsets to BaseCoord tables, from beginning of
				 * BaseValues table — order matches baselineTags array in
				 * the BaseTagList */
  public:
  DEFINE_SIZE_ARRAY (4, baseCoords);
};

struct BaseLangSysRecord
{
  int cmp (hb_tag_t key) const { return baseLangSysTag.cmp (key); }

  bool has_data () const { return baseLangSysTag; }

  const MinMax &get_min_max () const { return this+minMax; }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  minMax.sanitize (c, this)));
  }

  protected:
  Tag		baseLangSysTag;	/* 4-byte language system identification tag */
  OffsetTo<MinMax>
		minMax;		/* Offset to MinMax table, from beginning
				 * of BaseScript table */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseScript
{
  const MinMax &get_min_max (hb_tag_t language_tag) const
  {
    const BaseLangSysRecord& record = baseLangSysRecords.bsearch (language_tag);
    return record.has_data () ? record.get_min_max () : this+defaultMinMax;
  }

  const BaseCoord &get_base_coord (int baseline_tag_index) const
  { return (this+baseValues).get_base_coord (baseline_tag_index); }

  bool has_data () const { return baseValues; }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  baseValues.sanitize (c, this) &&
			  defaultMinMax.sanitize (c, this) &&
			  baseLangSysRecords.sanitize (c, this)));
  }

  protected:
  OffsetTo<BaseValues>
		baseValues;	/* Offset to BaseValues table, from beginning
				 * of BaseScript table (may be NULL) */
  OffsetTo<MinMax>
		defaultMinMax;	/* Offset to MinMax table, from beginning of
				 * BaseScript table (may be NULL) */
  SortedArrayOf<BaseLangSysRecord>
		baseLangSysRecords;
				/* Number of BaseLangSysRecords
				 * defined — may be zero (0) */

  public:
  DEFINE_SIZE_ARRAY (6, baseLangSysRecords);
};

struct BaseScriptList;
struct BaseScriptRecord
{
  int cmp (hb_tag_t key) const { return baseScriptTag.cmp (key); }

  bool has_data () const { return baseScriptTag; }

  const BaseScript &get_base_script (const BaseScriptList *list) const
  { return list+baseScript; }

  bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  baseScript.sanitize (c, base)));
  }

  protected:
  Tag		baseScriptTag;	/* 4-byte script identification tag */
  OffsetTo<BaseScript>
		baseScript;	/* Offset to BaseScript table, from beginning
				 * of BaseScriptList */

  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseScriptList
{
  const BaseScript &get_base_script (hb_tag_t script) const
  {
    const BaseScriptRecord *record = &baseScriptRecords.bsearch (script);
    if (!record->has_data ()) record = &baseScriptRecords.bsearch (HB_TAG ('D','F','L','T'));
    return record->has_data () ? record->get_base_script (this) : Null (BaseScript);
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseScriptRecords.sanitize (c, this));
  }

  protected:
  SortedArrayOf<BaseScriptRecord>
			baseScriptRecords;

  public:
  DEFINE_SIZE_ARRAY (2, baseScriptRecords);
};

struct Axis
{
  bool get_baseline (hb_tag_t          baseline_tag,
		     hb_tag_t          script_tag,
		     hb_tag_t          language_tag,
		     const BaseCoord **coord) const
  {
    const BaseScript &base_script = (this+baseScriptList).get_base_script (script_tag);
    if (!base_script.has_data ())
    {
      *coord = nullptr;
      return false;
    }

    if (likely (coord))
    {
      unsigned int tag_index = 0;
      if (!(this+baseTagList).bfind (baseline_tag, &tag_index))
      {
        *coord = nullptr;
        return false;
      }
      *coord = &base_script.get_base_coord (tag_index);
    }

    return true;
  }

  bool get_min_max (hb_tag_t          script_tag,
		    hb_tag_t          language_tag,
		    hb_tag_t          feature_tag,
		    const BaseCoord **min_coord,
		    const BaseCoord **max_coord) const
  {
    const BaseScript &base_script = (this+baseScriptList).get_base_script (script_tag);
    if (!base_script.has_data ())
    {
      *min_coord = *max_coord = nullptr;
      return false;
    }

    base_script.get_min_max (language_tag).get_min_max (feature_tag, min_coord, max_coord);

    return true;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  (this+baseTagList).sanitize (c) &&
			  (this+baseScriptList).sanitize (c)));
  }

  protected:
  OffsetTo<SortedArrayOf<Tag>>
		baseTagList;	/* Offset to BaseTagList table, from beginning
				 * of Axis table (may be NULL)
				 * Array of 4-byte baseline identification tags — must
				 * be in alphabetical order */
  OffsetTo<BaseScriptList>
		baseScriptList;	/* Offset to BaseScriptList table, from beginning
				 * of Axis table
				 * Array of BaseScriptRecords, in alphabetical order
				 * by baseScriptTag */

  public:
  DEFINE_SIZE_STATIC (4);
};

struct BASE
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_BASE;

  const Axis &get_axis (hb_direction_t direction) const
  { return HB_DIRECTION_IS_VERTICAL (direction) ? this+vAxis : this+hAxis; }

  const VariationStore &get_var_store () const
  { return version.to_int () < 0x00010001u ? Null (VariationStore) : this+varStore; }

  bool get_baseline (hb_font_t      *font,
		     hb_tag_t        baseline_tag,
		     hb_direction_t  direction,
		     hb_tag_t        script_tag,
		     hb_tag_t        language_tag,
		     hb_position_t  *base) const
  {
    const BaseCoord *base_coord = nullptr;
    if (unlikely (!get_axis (direction).get_baseline (baseline_tag, script_tag, language_tag, &base_coord) ||
		  !base_coord || !base_coord->has_data ()))
      return false;

    if (likely (base))
      *base = base_coord->get_coord (font, get_var_store (), direction);

    return true;
  }

  /* TODO: Expose this separately sometime? */
  bool get_min_max (hb_font_t      *font,
		    hb_direction_t  direction,
		    hb_tag_t        script_tag,
		    hb_tag_t        language_tag,
		    hb_tag_t        feature_tag,
		    hb_position_t  *min,
		    hb_position_t  *max)
  {
    const BaseCoord *min_coord, *max_coord;
    if (!get_axis (direction).get_min_max (script_tag, language_tag, feature_tag,
					   &min_coord, &max_coord))
      return false;

    const VariationStore &var_store = get_var_store ();
    if (likely (min && min_coord)) *min = min_coord->get_coord (font, var_store, direction);
    if (likely (max && max_coord)) *max = max_coord->get_coord (font, var_store, direction);
    return true;
  }

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (likely (c->check_struct (this) &&
			  likely (version.major == 1) &&
			  hAxis.sanitize (c, this) &&
			  vAxis.sanitize (c, this) &&
			  (version.to_int () < 0x00010001u || varStore.sanitize (c, this))));
  }

  protected:
  FixedVersion<>version;	/* Version of the BASE table */
  OffsetTo<Axis>hAxis;		/* Offset to horizontal Axis table, from beginning
				 * of BASE table (may be NULL) */
  OffsetTo<Axis>vAxis;		/* Offset to vertical Axis table, from beginning
				 * of BASE table (may be NULL) */
  LOffsetTo<VariationStore>
		varStore;	/* Offset to the table of Item Variation
				 * Store--from beginning of BASE
				 * header (may be NULL).  Introduced
				 * in version 0x00010001. */
  public:
  DEFINE_SIZE_MIN (8);
};


} /* namespace OT */


#endif /* HB_OT_LAYOUT_BASE_TABLE_HH */
