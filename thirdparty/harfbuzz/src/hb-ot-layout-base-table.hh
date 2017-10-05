/*
 * Copyright © 2016 Elie Roux <elie.roux@telecom-bretagne.eu>
 * Copyright © 2018  Google, Inc.
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

#include "hb-open-type-private.hh"
#include "hb-ot-layout-common-private.hh"

namespace OT {

/*
 * BASE -- Baseline
 * https://docs.microsoft.com/en-us/typography/opentype/spec/base
 */


/* XXX Review this. */
#define NOT_INDEXED		((unsigned int) -1)


struct BaseCoordFormat1
{
  inline int get_coord (void) const { return coordinate; }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 1 */
  FWORD		coordinate;	/* X or Y value, in design units */
  public:
  DEFINE_SIZE_STATIC (4);
};

struct BaseCoordFormat2
{
  inline int get_coord (void) const
  {
    /* TODO */
    return coordinate;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;		/* Format identifier--format = 2 */
  FWORD		coordinate;	/* X or Y value, in design units */
  GlyphID	referenceGlyph;	/* Glyph ID of control glyph */
  HBUINT16	coordPoint;	/* Index of contour point on the
				 * reference glyph */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct BaseCoordFormat3
{
  inline int get_coord (void) const
  {
    /* TODO */
    return coordinate;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && deviceTable.sanitize (c, this));
  }

  protected:
  HBUINT16		format;		/* Format identifier--format = 3 */
  FWORD			coordinate;	/* X or Y value, in design units */
  OffsetTo<Device>	deviceTable;	/* Offset to Device table for X or
					 * Y value, from beginning of
					 * BaseCoord table (may be NULL). */
  public:
  DEFINE_SIZE_STATIC (6);
};

struct BaseCoord
{
  inline int get_coord (void) const
  {
    /* XXX wire up direction and font. */
    switch (u.format) {
    case 1: return u.format1.get_coord ();
    case 2: return u.format2.get_coord ();
    case 3: return u.format3.get_coord ();
    default:return 0;
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
  inline int get_min_value (void) const { return (this+minCoord).get_coord(); }
  inline int get_max_value (void) const { return (this+maxCoord).get_coord(); }

  inline const Tag& get_tag () const { return tag; }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  minCoord.sanitize (c, base) &&
		  maxCoord.sanitize (c, base));
  }

  protected:
  Tag                   tag;		/* 4-byte feature identification tag--must
					 * match feature tag in FeatureList */
  OffsetTo<BaseCoord>   minCoord;	/* Offset to BaseCoord table that defines
					 * the minimum extent value, from beginning
					 * of MinMax table (may be NULL) */
  OffsetTo<BaseCoord>   maxCoord;	/* Offset to BaseCoord table that defines
					 * the maximum extent value, from beginning
					 * of MinMax table (may be NULL) */
  public:
  DEFINE_SIZE_STATIC (8);

};

struct MinMax
{
  inline unsigned int get_feature_tag_index (Tag featureTableTag) const
  {
    /* TODO bsearch */
    unsigned int count = featMinMaxRecords.len;
    for (unsigned int i = 0; i < count; i++)
    {
      Tag tag = featMinMaxRecords[i].get_tag ();
      int cmp = tag.cmp(featureTableTag);
      if (cmp == 0) return i;
      if (cmp > 0)  return NOT_INDEXED;
    }
    return NOT_INDEXED;
  }

  inline int get_min_value (unsigned int featureTableTagIndex) const
  {
    if (featureTableTagIndex == NOT_INDEXED)
      return (this+minCoord).get_coord();
    return featMinMaxRecords[featureTableTagIndex].get_min_value();
  }

  inline int get_max_value (unsigned int featureTableTagIndex) const
  {
    if (featureTableTagIndex == NOT_INDEXED)
      return (this+maxCoord).get_coord();
    return featMinMaxRecords[featureTableTagIndex].get_max_value();
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  minCoord.sanitize (c, this) &&
		  maxCoord.sanitize (c, this) &&
		  featMinMaxRecords.sanitize (c, this));
  }

  protected:
  OffsetTo<BaseCoord>	minCoord;	/* Offset to BaseCoord table that defines
					 * minimum extent value, from the beginning
					 * of MinMax table (may be NULL) */
  OffsetTo<BaseCoord>	maxCoord;	/* Offset to BaseCoord table that defines
					 * maximum extent value, from the beginning
					 * of MinMax table (may be NULL) */
  ArrayOf<FeatMinMaxRecord>
		featMinMaxRecords;	/* Array of FeatMinMaxRecords, in alphabetical
					 * order by featureTableTag */
  public:
  DEFINE_SIZE_ARRAY (6, featMinMaxRecords);
};

/* TODO... */
struct BaseLangSysRecord
{
  inline const Tag& get_tag(void) const
  { return baseLangSysTag; }

  inline unsigned int get_feature_tag_index (Tag featureTableTag) const
  { return (this+minMax).get_feature_tag_index( featureTableTag); }

  inline int get_min_value (unsigned int featureTableTagIndex) const
  { return (this+minMax).get_min_value( featureTableTagIndex); }

  inline int get_max_value (unsigned int featureTableTagIndex) const
  { return (this+minMax).get_max_value (featureTableTagIndex); }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  minMax.sanitize (c, base));
  }

  protected:
  Tag			baseLangSysTag;
  OffsetTo<MinMax>	minMax;
  public:
  DEFINE_SIZE_STATIC (6);

};

struct BaseValues
{
  inline unsigned int get_default_base_tag_index (void) const
  { return defaultIndex; }

  inline int get_base_coord (unsigned int baselineTagIndex) const
  {
    return (this+baseCoords[baselineTagIndex]).get_coord ();
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseCoords.sanitize (c, this));
  }

  protected:
  Index				defaultIndex;
  OffsetArrayOf<BaseCoord>	baseCoords;
  public:
  DEFINE_SIZE_ARRAY (4, baseCoords);
};

struct BaseScript {

  inline unsigned int get_lang_tag_index (Tag baseLangSysTag) const
  {
    /* XXX bsearch */
    Tag tag;
    int cmp;
    unsigned int count = baseLangSysRecords.len;
    for (unsigned int i = 0; i < count; i++)
    {
      tag = baseLangSysRecords[i].get_tag ();
      // taking advantage of alphabetical order
      cmp = tag.cmp(baseLangSysTag);
      if (cmp == 0) return i;
      if (cmp > 0)  return NOT_INDEXED;
    }
    return NOT_INDEXED;
  }

  inline unsigned int get_feature_tag_index (unsigned int baseLangSysIndex, Tag featureTableTag) const
  {
    if (baseLangSysIndex == NOT_INDEXED)
    {
      if (unlikely(defaultMinMax)) return NOT_INDEXED;
      return (this+defaultMinMax).get_feature_tag_index (featureTableTag);
    }
    return baseLangSysRecords[baseLangSysIndex].get_feature_tag_index (featureTableTag);
  }

  inline int get_min_value (unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    if (baseLangSysIndex == NOT_INDEXED)
      return (this+defaultMinMax).get_min_value (featureTableTagIndex);
    return baseLangSysRecords[baseLangSysIndex].get_max_value (featureTableTagIndex);
  }

  inline int get_max_value (unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    if (baseLangSysIndex == NOT_INDEXED)
      return (this+defaultMinMax).get_min_value (featureTableTagIndex);
    return baseLangSysRecords[baseLangSysIndex].get_max_value (featureTableTagIndex);
  }

  inline unsigned int get_default_base_tag_index (void) const
  { return (this+baseValues).get_default_base_tag_index (); }

  inline int get_base_coord (unsigned int baselineTagIndex) const
  { return (this+baseValues).get_base_coord (baselineTagIndex); }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseValues.sanitize (c, this) &&
		  defaultMinMax.sanitize (c, this) &&
		  baseLangSysRecords.sanitize (c, this));
  }

  protected:
  OffsetTo<BaseValues>		baseValues;
  OffsetTo<MinMax>		defaultMinMax;
  ArrayOf<BaseLangSysRecord>	baseLangSysRecords;

  public:
    DEFINE_SIZE_ARRAY (6, baseLangSysRecords);
};


struct BaseScriptRecord {

  inline const Tag& get_tag (void) const
  { return baseScriptTag; }

  inline unsigned int get_default_base_tag_index(void) const
  { return (this+baseScript).get_default_base_tag_index (); }

  inline int get_base_coord(unsigned int baselineTagIndex) const
  { return (this+baseScript).get_base_coord (baselineTagIndex); }

  inline unsigned int get_lang_tag_index (Tag baseLangSysTag) const
  { return (this+baseScript).get_lang_tag_index (baseLangSysTag); }

  inline unsigned int get_feature_tag_index (unsigned int baseLangSysIndex, Tag featureTableTag) const
  { return (this+baseScript).get_feature_tag_index (baseLangSysIndex, featureTableTag); }

  inline int get_max_value (unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  { return (this+baseScript).get_max_value (baseLangSysIndex, featureTableTagIndex); }

  inline int get_min_value (unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  { return (this+baseScript).get_min_value (baseLangSysIndex, featureTableTagIndex); }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseScript.sanitize (c, base));
  }

  protected:
  Tag                   baseScriptTag;
  OffsetTo<BaseScript>  baseScript;

  public:
    DEFINE_SIZE_STATIC (6);
};

struct BaseScriptList {

  inline unsigned int get_base_script_index (Tag baseScriptTag) const
  {
    /* XXX bsearch? */
    unsigned int count = baseScriptRecords.len;
    for (unsigned int i = 0; i < count; i++)
      if (baseScriptRecords[i].get_tag() == baseScriptTag)
        return i;
    return NOT_INDEXED;
  }

  inline unsigned int get_default_base_tag_index (unsigned int baseScriptIndex) const
  {
    return baseScriptRecords[baseScriptIndex].get_default_base_tag_index();
  }

  inline int get_base_coord(unsigned int baseScriptIndex, unsigned int baselineTagIndex) const
  {
    return baseScriptRecords[baseScriptIndex].get_base_coord(baselineTagIndex);
  }

  inline unsigned int get_lang_tag_index (unsigned int baseScriptIndex, Tag baseLangSysTag) const
  {
    return baseScriptRecords[baseScriptIndex].get_lang_tag_index(baseLangSysTag);
  }

  inline unsigned int get_feature_tag_index (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, Tag featureTableTag) const
  {
    return baseScriptRecords[baseScriptIndex].get_feature_tag_index(baseLangSysIndex, featureTableTag);
  }

  inline int get_max_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return baseScriptRecords[baseScriptIndex].get_max_value(baseLangSysIndex, featureTableTagIndex);
  }

  inline int get_min_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return baseScriptRecords[baseScriptIndex].get_min_value(baseLangSysIndex, featureTableTagIndex);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseScriptRecords.sanitize (c, this));
  }

  protected:
  ArrayOf<BaseScriptRecord>	baseScriptRecords;

  public:
  DEFINE_SIZE_ARRAY (2, baseScriptRecords);
};

struct BaseTagList
{
  inline unsigned int get_tag_index (Tag baselineTag) const
  {
    /* TODO bsearch? */
    unsigned int count = baselineTags.len;
    for (unsigned int i = 0; i < count; i++)
      if (baselineTags[i] == baselineTag)
        return i;
    return NOT_INDEXED;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  SortedArrayOf<Tag>  baselineTags;

  public:
  DEFINE_SIZE_ARRAY (2, baselineTags);
};

struct Axis
{

  inline unsigned int get_base_tag_index (Tag baselineTag) const
  {
    return (this+baseTagList).get_tag_index(baselineTag);
  }

  inline unsigned int get_default_base_tag_index_for_script_index (unsigned int baseScriptIndex) const
  {
    return (this+baseScriptList).get_default_base_tag_index(baseScriptIndex);
  }

  inline int get_base_coord (unsigned int baseScriptIndex, unsigned int baselineTagIndex) const
  {
    return (this+baseScriptList).get_base_coord(baseScriptIndex, baselineTagIndex);
  }

  inline unsigned int get_lang_tag_index (unsigned int baseScriptIndex, Tag baseLangSysTag) const
  {
    return (this+baseScriptList).get_lang_tag_index(baseScriptIndex, baseLangSysTag);
  }

  inline unsigned int get_feature_tag_index (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, Tag featureTableTag) const
  {
    return (this+baseScriptList).get_feature_tag_index(baseScriptIndex, baseLangSysIndex, featureTableTag);
  }

  inline int get_max_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+baseScriptList).get_max_value(baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline int get_min_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+baseScriptList).get_min_value(baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  baseTagList.sanitize (c, this) &&
		  baseScriptList.sanitize (c, this));
  }

  protected:
  OffsetTo<BaseTagList>		baseTagList;
  OffsetTo<BaseScriptList>	baseScriptList;

  public:
  DEFINE_SIZE_STATIC (4);
};

struct BASE
{
  static const hb_tag_t tableTag = HB_OT_TAG_BASE;

  inline bool has_v_axis(void) { return vAxis != 0; }

  inline bool has_h_axis(void) { return hAxis != 0; }

  inline unsigned int get_h_base_tag_index (Tag baselineTag) const
  {
    return (this+hAxis).get_base_tag_index(baselineTag);
  }

  inline unsigned int get_h_default_base_tag_index_for_script_index (unsigned int baseScriptIndex) const
  {
    return (this+hAxis).get_default_base_tag_index_for_script_index(baseScriptIndex);
  }

  inline int get_h_base_coord(unsigned int baseScriptIndex, unsigned int baselineTagIndex) const
  {
    return (this+hAxis).get_base_coord(baseScriptIndex, baselineTagIndex);
  }

  inline unsigned int get_v_base_tag_index(Tag baselineTag) const
  {
    return (this+vAxis).get_base_tag_index(baselineTag);
  }

  inline unsigned int get_v_default_base_tag_index_for_script_index (unsigned int baseScriptIndex) const
  {
    return (this+vAxis).get_default_base_tag_index_for_script_index(baseScriptIndex);
  }

  inline int get_v_base_coord(unsigned int baseScriptIndex, unsigned int baselineTagIndex) const
  {
    return (this+vAxis).get_base_coord(baseScriptIndex, baselineTagIndex);
  }

  inline unsigned int get_h_lang_tag_index (unsigned int baseScriptIndex, Tag baseLangSysTag) const
  {
    return (this+hAxis).get_lang_tag_index (baseScriptIndex, baseLangSysTag);
  }

  inline unsigned int get_h_feature_tag_index (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, Tag featureTableTag) const
  {
    return (this+hAxis).get_feature_tag_index (baseScriptIndex, baseLangSysIndex, featureTableTag);
  }

  inline int get_h_max_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+hAxis).get_max_value (baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline int get_h_min_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+hAxis).get_min_value (baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline unsigned int get_v_lang_tag_index (unsigned int baseScriptIndex, Tag baseLangSysTag) const
  {
    return (this+vAxis).get_lang_tag_index (baseScriptIndex, baseLangSysTag);
  }

  inline unsigned int get_v_feature_tag_index (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, Tag featureTableTag) const
  {
    return (this+vAxis).get_feature_tag_index (baseScriptIndex, baseLangSysIndex, featureTableTag);
  }

  inline int get_v_max_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+vAxis).get_max_value (baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline int get_v_min_value (unsigned int baseScriptIndex, unsigned int baseLangSysIndex, unsigned int featureTableTagIndex) const
  {
    return (this+vAxis).get_min_value (baseScriptIndex, baseLangSysIndex, featureTableTagIndex);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version.major == 1) &&
		  hAxis.sanitize (c, this) &&
		  vAxis.sanitize (c, this) &&
		  (version.to_int () < 0x00010001u || varStore.sanitize (c, this)));
  }

  protected:
  FixedVersion<>  version;
  OffsetTo<Axis>  hAxis;
  OffsetTo<Axis>  vAxis;
  LOffsetTo<VariationStore>
		varStore;		/* Offset to the table of Item Variation
					 * Store--from beginning of BASE
					 * header (may be NULL).  Introduced
					 * in version 0x00010001. */
  public:
  DEFINE_SIZE_MIN (8);
};


} /* namespace OT */


#endif /* HB_OT_LAYOUT_BASE_TABLE_HH */
