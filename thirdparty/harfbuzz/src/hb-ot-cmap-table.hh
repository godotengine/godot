/*
 * Copyright © 2014  Google, Inc.
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

#ifndef HB_OT_CMAP_TABLE_HH
#define HB_OT_CMAP_TABLE_HH

#include "hb-open-type-private.hh"


namespace OT {


/*
 * cmap -- Character To Glyph Index Mapping Table
 */

#define HB_OT_TAG_cmap HB_TAG('c','m','a','p')


struct CmapSubtableFormat0
{
  inline bool get_glyph (hb_codepoint_t codepoint, hb_codepoint_t *glyph) const
  {
    hb_codepoint_t gid = codepoint < 256 ? glyphIdArray[codepoint] : 0;
    if (!gid)
      return false;
    *glyph = gid;
    return true;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  protected:
  HBUINT16	format;		/* Format number is set to 0. */
  HBUINT16	lengthZ;	/* Byte length of this subtable. */
  HBUINT16	languageZ;	/* Ignore. */
  HBUINT8		glyphIdArray[256];/* An array that maps character
				 * code to glyph index values. */
  public:
  DEFINE_SIZE_STATIC (6 + 256);
};

struct CmapSubtableFormat4
{
  struct accelerator_t
  {
    inline void init (const CmapSubtableFormat4 *subtable)
    {
      segCount = subtable->segCountX2 / 2;
      endCount = subtable->values;
      startCount = endCount + segCount + 1;
      idDelta = startCount + segCount;
      idRangeOffset = idDelta + segCount;
      glyphIdArray = idRangeOffset + segCount;
      glyphIdArrayLength = (subtable->length - 16 - 8 * segCount) / 2;
    }

    static inline bool get_glyph_func (const void *obj, hb_codepoint_t codepoint, hb_codepoint_t *glyph)
    {
      const accelerator_t *thiz = (const accelerator_t *) obj;

      /* Custom two-array bsearch. */
      int min = 0, max = (int) thiz->segCount - 1;
      const HBUINT16 *startCount = thiz->startCount;
      const HBUINT16 *endCount = thiz->endCount;
      unsigned int i;
      while (min <= max)
      {
	int mid = (min + max) / 2;
	if (codepoint < startCount[mid])
	  max = mid - 1;
	else if (codepoint > endCount[mid])
	  min = mid + 1;
	else
	{
	  i = mid;
	  goto found;
	}
      }
      return false;

    found:
      hb_codepoint_t gid;
      unsigned int rangeOffset = thiz->idRangeOffset[i];
      if (rangeOffset == 0)
	gid = codepoint + thiz->idDelta[i];
      else
      {
	/* Somebody has been smoking... */
	unsigned int index = rangeOffset / 2 + (codepoint - thiz->startCount[i]) + i - thiz->segCount;
	if (unlikely (index >= thiz->glyphIdArrayLength))
	  return false;
	gid = thiz->glyphIdArray[index];
	if (unlikely (!gid))
	  return false;
	gid += thiz->idDelta[i];
      }

      *glyph = gid & 0xFFFFu;
      return true;
    }

    const HBUINT16 *endCount;
    const HBUINT16 *startCount;
    const HBUINT16 *idDelta;
    const HBUINT16 *idRangeOffset;
    const HBUINT16 *glyphIdArray;
    unsigned int segCount;
    unsigned int glyphIdArrayLength;
  };

  inline bool get_glyph (hb_codepoint_t codepoint, hb_codepoint_t *glyph) const
  {
    accelerator_t accel;
    accel.init (this);
    return accel.get_glyph_func (&accel, codepoint, glyph);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    if (unlikely (!c->check_range (this, length)))
    {
      /* Some broken fonts have too long of a "length" value.
       * If that is the case, just change the value to truncate
       * the subtable at the end of the blob. */
      uint16_t new_length = (uint16_t) MIN ((uintptr_t) 65535,
					    (uintptr_t) (c->end -
							 (char *) this));
      if (!c->try_set (&length, new_length))
	return_trace (false);
    }

    return_trace (16 + 4 * (unsigned int) segCountX2 <= length);
  }

  protected:
  HBUINT16	format;		/* Format number is set to 4. */
  HBUINT16	length;		/* This is the length in bytes of the
				 * subtable. */
  HBUINT16	languageZ;	/* Ignore. */
  HBUINT16	segCountX2;	/* 2 x segCount. */
  HBUINT16	searchRangeZ;	/* 2 * (2**floor(log2(segCount))) */
  HBUINT16	entrySelectorZ;	/* log2(searchRange/2) */
  HBUINT16	rangeShiftZ;	/* 2 x segCount - searchRange */

  HBUINT16	values[VAR];
#if 0
  HBUINT16	endCount[segCount];	/* End characterCode for each segment,
					 * last=0xFFFFu. */
  HBUINT16	reservedPad;		/* Set to 0. */
  HBUINT16	startCount[segCount];	/* Start character code for each segment. */
  HBINT16		idDelta[segCount];	/* Delta for all character codes in segment. */
  HBUINT16	idRangeOffset[segCount];/* Offsets into glyphIdArray or 0 */
  HBUINT16	glyphIdArray[VAR];	/* Glyph index array (arbitrary length) */
#endif

  public:
  DEFINE_SIZE_ARRAY (14, values);
};

struct CmapSubtableLongGroup
{
  friend struct CmapSubtableFormat12;
  friend struct CmapSubtableFormat13;

  int cmp (hb_codepoint_t codepoint) const
  {
    if (codepoint < startCharCode) return -1;
    if (codepoint > endCharCode)   return +1;
    return 0;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  private:
  HBUINT32		startCharCode;	/* First character code in this group. */
  HBUINT32		endCharCode;	/* Last character code in this group. */
  HBUINT32		glyphID;	/* Glyph index; interpretation depends on
				 * subtable format. */
  public:
  DEFINE_SIZE_STATIC (12);
};

template <typename UINT>
struct CmapSubtableTrimmed
{
  inline bool get_glyph (hb_codepoint_t codepoint, hb_codepoint_t *glyph) const
  {
    /* Rely on our implicit array bound-checking. */
    hb_codepoint_t gid = glyphIdArray[codepoint - startCharCode];
    if (!gid)
      return false;
    *glyph = gid;
    return true;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && glyphIdArray.sanitize (c));
  }

  protected:
  UINT		formatReserved;	/* Subtable format and (maybe) padding. */
  UINT		lengthZ;	/* Byte length of this subtable. */
  UINT		languageZ;	/* Ignore. */
  UINT		startCharCode;	/* First character code covered. */
  ArrayOf<GlyphID, UINT>
		glyphIdArray;	/* Array of glyph index values for character
				 * codes in the range. */
  public:
  DEFINE_SIZE_ARRAY (5 * sizeof (UINT), glyphIdArray);
};

struct CmapSubtableFormat6  : CmapSubtableTrimmed<HBUINT16> {};
struct CmapSubtableFormat10 : CmapSubtableTrimmed<HBUINT32 > {};

template <typename T>
struct CmapSubtableLongSegmented
{
  inline bool get_glyph (hb_codepoint_t codepoint, hb_codepoint_t *glyph) const
  {
    int i = groups.bsearch (codepoint);
    if (i == -1)
      return false;
    *glyph = T::group_get_glyph (groups[i], codepoint);
    return true;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && groups.sanitize (c));
  }

  protected:
  HBUINT16	format;		/* Subtable format; set to 12. */
  HBUINT16	reservedZ;	/* Reserved; set to 0. */
  HBUINT32		lengthZ;	/* Byte length of this subtable. */
  HBUINT32		languageZ;	/* Ignore. */
  SortedArrayOf<CmapSubtableLongGroup, HBUINT32>
		groups;		/* Groupings. */
  public:
  DEFINE_SIZE_ARRAY (16, groups);
};

struct CmapSubtableFormat12 : CmapSubtableLongSegmented<CmapSubtableFormat12>
{
  static inline hb_codepoint_t group_get_glyph (const CmapSubtableLongGroup &group,
						hb_codepoint_t u)
  { return group.glyphID + (u - group.startCharCode); }
};

struct CmapSubtableFormat13 : CmapSubtableLongSegmented<CmapSubtableFormat13>
{
  static inline hb_codepoint_t group_get_glyph (const CmapSubtableLongGroup &group,
						hb_codepoint_t u HB_UNUSED)
  { return group.glyphID; }
};

typedef enum
{
  GLYPH_VARIANT_NOT_FOUND = 0,
  GLYPH_VARIANT_FOUND = 1,
  GLYPH_VARIANT_USE_DEFAULT = 2
} glyph_variant_t;

struct UnicodeValueRange
{
  inline int cmp (const hb_codepoint_t &codepoint) const
  {
    if (codepoint < startUnicodeValue) return -1;
    if (codepoint > startUnicodeValue + additionalCount) return +1;
    return 0;
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  UINT24	startUnicodeValue;	/* First value in this range. */
  HBUINT8		additionalCount;	/* Number of additional values in this
					 * range. */
  public:
  DEFINE_SIZE_STATIC (4);
};

typedef SortedArrayOf<UnicodeValueRange, HBUINT32> DefaultUVS;

struct UVSMapping
{
  inline int cmp (const hb_codepoint_t &codepoint) const
  {
    return unicodeValue.cmp (codepoint);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  UINT24	unicodeValue;	/* Base Unicode value of the UVS */
  GlyphID	glyphID;	/* Glyph ID of the UVS */
  public:
  DEFINE_SIZE_STATIC (5);
};

typedef SortedArrayOf<UVSMapping, HBUINT32> NonDefaultUVS;

struct VariationSelectorRecord
{
  inline glyph_variant_t get_glyph (hb_codepoint_t codepoint,
				    hb_codepoint_t *glyph,
				    const void *base) const
  {
    int i;
    const DefaultUVS &defaults = base+defaultUVS;
    i = defaults.bsearch (codepoint);
    if (i != -1)
      return GLYPH_VARIANT_USE_DEFAULT;
    const NonDefaultUVS &nonDefaults = base+nonDefaultUVS;
    i = nonDefaults.bsearch (codepoint);
    if (i != -1)
    {
      *glyph = nonDefaults[i].glyphID;
       return GLYPH_VARIANT_FOUND;
    }
    return GLYPH_VARIANT_NOT_FOUND;
  }

  inline int cmp (const hb_codepoint_t &variation_selector) const
  {
    return varSelector.cmp (variation_selector);
  }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  defaultUVS.sanitize (c, base) &&
		  nonDefaultUVS.sanitize (c, base));
  }

  UINT24	varSelector;	/* Variation selector. */
  LOffsetTo<DefaultUVS>
		defaultUVS;	/* Offset to Default UVS Table. May be 0. */
  LOffsetTo<NonDefaultUVS>
		nonDefaultUVS;	/* Offset to Non-Default UVS Table. May be 0. */
  public:
  DEFINE_SIZE_STATIC (11);
};

struct CmapSubtableFormat14
{
  inline glyph_variant_t get_glyph_variant (hb_codepoint_t codepoint,
					    hb_codepoint_t variation_selector,
					    hb_codepoint_t *glyph) const
  {
    return record[record.bsearch(variation_selector)].get_glyph (codepoint, glyph, this);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  record.sanitize (c, this));
  }

  protected:
  HBUINT16	format;		/* Format number is set to 14. */
  HBUINT32		lengthZ;	/* Byte length of this subtable. */
  SortedArrayOf<VariationSelectorRecord, HBUINT32>
		record;		/* Variation selector records; sorted
				 * in increasing order of `varSelector'. */
  public:
  DEFINE_SIZE_ARRAY (10, record);
};

struct CmapSubtable
{
  /* Note: We intentionally do NOT implement subtable formats 2 and 8. */

  inline bool get_glyph (hb_codepoint_t codepoint,
			 hb_codepoint_t *glyph) const
  {
    switch (u.format) {
    case  0: return u.format0 .get_glyph(codepoint, glyph);
    case  4: return u.format4 .get_glyph(codepoint, glyph);
    case  6: return u.format6 .get_glyph(codepoint, glyph);
    case 10: return u.format10.get_glyph(codepoint, glyph);
    case 12: return u.format12.get_glyph(codepoint, glyph);
    case 13: return u.format13.get_glyph(codepoint, glyph);
    case 14:
    default: return false;
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (!u.format.sanitize (c)) return_trace (false);
    switch (u.format) {
    case  0: return_trace (u.format0 .sanitize (c));
    case  4: return_trace (u.format4 .sanitize (c));
    case  6: return_trace (u.format6 .sanitize (c));
    case 10: return_trace (u.format10.sanitize (c));
    case 12: return_trace (u.format12.sanitize (c));
    case 13: return_trace (u.format13.sanitize (c));
    case 14: return_trace (u.format14.sanitize (c));
    default:return_trace (true);
    }
  }

  public:
  union {
  HBUINT16		format;		/* Format identifier */
  CmapSubtableFormat0	format0;
  CmapSubtableFormat4	format4;
  CmapSubtableFormat6	format6;
  CmapSubtableFormat10	format10;
  CmapSubtableFormat12	format12;
  CmapSubtableFormat13	format13;
  CmapSubtableFormat14	format14;
  } u;
  public:
  DEFINE_SIZE_UNION (2, format);
};


struct EncodingRecord
{
  inline int cmp (const EncodingRecord &other) const
  {
    int ret;
    ret = platformID.cmp (other.platformID);
    if (ret) return ret;
    ret = encodingID.cmp (other.encodingID);
    if (ret) return ret;
    return 0;
  }

  inline bool sanitize (hb_sanitize_context_t *c, const void *base) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  subtable.sanitize (c, base));
  }

  HBUINT16	platformID;	/* Platform ID. */
  HBUINT16	encodingID;	/* Platform-specific encoding ID. */
  LOffsetTo<CmapSubtable>
		subtable;	/* Byte offset from beginning of table to the subtable for this encoding. */
  public:
  DEFINE_SIZE_STATIC (8);
};

struct cmap
{
  static const hb_tag_t tableTag	= HB_OT_TAG_cmap;

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version == 0) &&
		  encodingRecord.sanitize (c, this));
  }

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      this->blob = OT::Sanitizer<OT::cmap>().sanitize (face->reference_table (HB_OT_TAG_cmap));
      const OT::cmap *cmap = OT::Sanitizer<OT::cmap>::lock_instance (this->blob);
      const OT::CmapSubtable *subtable = nullptr;
      const OT::CmapSubtableFormat14 *subtable_uvs = nullptr;

      bool symbol = false;
      /* 32-bit subtables. */
      if (!subtable) subtable = cmap->find_subtable (3, 10);
      if (!subtable) subtable = cmap->find_subtable (0, 6);
      if (!subtable) subtable = cmap->find_subtable (0, 4);
      /* 16-bit subtables. */
      if (!subtable) subtable = cmap->find_subtable (3, 1);
      if (!subtable) subtable = cmap->find_subtable (0, 3);
      if (!subtable) subtable = cmap->find_subtable (0, 2);
      if (!subtable) subtable = cmap->find_subtable (0, 1);
      if (!subtable) subtable = cmap->find_subtable (0, 0);
      if (!subtable)
      {
	subtable = cmap->find_subtable (3, 0);
	if (subtable) symbol = true;
      }
      /* Meh. */
      if (!subtable) subtable = &OT::Null(OT::CmapSubtable);

      /* UVS subtable. */
      if (!subtable_uvs)
      {
	const OT::CmapSubtable *st = cmap->find_subtable (0, 5);
	if (st && st->u.format == 14)
	  subtable_uvs = &st->u.format14;
      }
      /* Meh. */
      if (!subtable_uvs) subtable_uvs = &OT::Null(OT::CmapSubtableFormat14);

      this->uvs_table = subtable_uvs;

      this->get_glyph_data = subtable;
      if (unlikely (symbol))
	this->get_glyph_func = get_glyph_from_symbol<OT::CmapSubtable>;
      else
	switch (subtable->u.format) {
	/* Accelerate format 4 and format 12. */
	default: this->get_glyph_func = get_glyph_from<OT::CmapSubtable>;		break;
	case 12: this->get_glyph_func = get_glyph_from<OT::CmapSubtableFormat12>;	break;
	case  4:
	  {
	    this->format4_accel.init (&subtable->u.format4);
	    this->get_glyph_data = &this->format4_accel;
	    this->get_glyph_func = this->format4_accel.get_glyph_func;
	  }
	  break;
	}
    }

    inline void fini (void)
    {
      hb_blob_destroy (this->blob);
    }

    inline bool get_nominal_glyph (hb_codepoint_t  unicode,
				   hb_codepoint_t *glyph) const
    {
      return this->get_glyph_func (this->get_glyph_data, unicode, glyph);
    }

    inline bool get_variation_glyph (hb_codepoint_t  unicode,
				     hb_codepoint_t  variation_selector,
				     hb_codepoint_t *glyph) const
    {
      switch (this->uvs_table->get_glyph_variant (unicode,
						  variation_selector,
						  glyph))
      {
	case OT::GLYPH_VARIANT_NOT_FOUND:		return false;
	case OT::GLYPH_VARIANT_FOUND:		return true;
	case OT::GLYPH_VARIANT_USE_DEFAULT:	break;
      }

      return get_nominal_glyph (unicode, glyph);
    }

    protected:
    typedef bool (*hb_cmap_get_glyph_func_t) (const void *obj,
					      hb_codepoint_t codepoint,
					      hb_codepoint_t *glyph);

    template <typename Type>
    static inline bool get_glyph_from (const void *obj,
				       hb_codepoint_t codepoint,
				       hb_codepoint_t *glyph)
    {
      const Type *typed_obj = (const Type *) obj;
      return typed_obj->get_glyph (codepoint, glyph);
    }

    template <typename Type>
    static inline bool get_glyph_from_symbol (const void *obj,
					      hb_codepoint_t codepoint,
					      hb_codepoint_t *glyph)
    {
      const Type *typed_obj = (const Type *) obj;
      if (likely (typed_obj->get_glyph (codepoint, glyph)))
	return true;

      if (codepoint <= 0x00FFu)
      {
	/* For symbol-encoded OpenType fonts, we duplicate the
	 * U+F000..F0FF range at U+0000..U+00FF.  That's what
	 * Windows seems to do, and that's hinted about at:
	 * http://www.microsoft.com/typography/otspec/recom.htm
	 * under "Non-Standard (Symbol) Fonts". */
	return typed_obj->get_glyph (0xF000u + codepoint, glyph);
      }

      return false;
    }

    private:
    hb_cmap_get_glyph_func_t get_glyph_func;
    const void *get_glyph_data;
    OT::CmapSubtableFormat4::accelerator_t format4_accel;

    const OT::CmapSubtableFormat14 *uvs_table;
    hb_blob_t *blob;
  };

  protected:

  inline const CmapSubtable *find_subtable (unsigned int platform_id,
					    unsigned int encoding_id) const
  {
    EncodingRecord key;
    key.platformID.set (platform_id);
    key.encodingID.set (encoding_id);

    /* Note: We can use bsearch, but since it has no performance
     * implications, we use lsearch and as such accept fonts with
     * unsorted subtable list. */
    int result = encodingRecord./*bsearch*/lsearch (key);
    if (result == -1 || !encodingRecord[result].subtable)
      return nullptr;

    return &(this+encodingRecord[result].subtable);
  }

  protected:
  HBUINT16		version;	/* Table version number (0). */
  SortedArrayOf<EncodingRecord>
			encodingRecord;	/* Encoding tables. */
  public:
  DEFINE_SIZE_ARRAY (4, encodingRecord);
};


} /* namespace OT */


#endif /* HB_OT_CMAP_TABLE_HH */
