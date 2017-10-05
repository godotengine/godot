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
#include "hb-set-private.hh"
#include "hb-subset-plan.hh"

/*
 * cmap -- Character to Glyph Index Mapping
 * https://docs.microsoft.com/en-us/typography/opentype/spec/cmap
 */
#define HB_OT_TAG_cmap HB_TAG('c','m','a','p')

#ifndef HB_MAX_UNICODE_CODEPOINT_VALUE
#define HB_MAX_UNICODE_CODEPOINT_VALUE 0x10FFFF
#endif

namespace OT {


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
  HBUINT16	length;		/* Byte length of this subtable. */
  HBUINT16	language;	/* Ignore. */
  HBUINT8	glyphIdArray[256];/* An array that maps character
				 * code to glyph index values. */
  public:
  DEFINE_SIZE_STATIC (6 + 256);
};

struct CmapSubtableFormat4
{
  struct segment_plan
  {
    HBUINT16 start_code;
    HBUINT16 end_code;
    bool use_delta;
  };

  bool serialize (hb_serialize_context_t *c,
                  const hb_subset_plan_t *plan,
                  const hb_vector_t<segment_plan> &segments)
  {
    TRACE_SERIALIZE (this);

    if (unlikely (!c->extend_min (*this))) return_trace (false);

    this->format.set (4);
    this->length.set (get_sub_table_size (segments));

    this->segCountX2.set (segments.len * 2);
    this->entrySelector.set (MAX (1u, hb_bit_storage (segments.len)) - 1);
    this->searchRange.set (2 * (1u << this->entrySelector));
    this->rangeShift.set (segments.len * 2 > this->searchRange
                          ? 2 * segments.len - this->searchRange
                          : 0);

    HBUINT16 *end_count = c->allocate_size<HBUINT16> (HBUINT16::static_size * segments.len);
    c->allocate_size<HBUINT16> (HBUINT16::static_size); // 2 bytes of padding.
    HBUINT16 *start_count = c->allocate_size<HBUINT16> (HBUINT16::static_size * segments.len);
    HBINT16 *id_delta = c->allocate_size<HBINT16> (HBUINT16::static_size * segments.len);
    HBUINT16 *id_range_offset = c->allocate_size<HBUINT16> (HBUINT16::static_size * segments.len);

    if (id_range_offset == nullptr)
      return_trace (false);

    for (unsigned int i = 0; i < segments.len; i++)
    {
      end_count[i].set (segments[i].end_code);
      start_count[i].set (segments[i].start_code);
      if (segments[i].use_delta)
      {
        hb_codepoint_t cp = segments[i].start_code;
        hb_codepoint_t start_gid = 0;
        if (unlikely (!plan->new_gid_for_codepoint (cp, &start_gid) && cp != 0xFFFF))
          return_trace (false);
        id_delta[i].set (start_gid - segments[i].start_code);
      } else {
        id_delta[i].set (0);
        unsigned int num_codepoints = segments[i].end_code - segments[i].start_code + 1;
        HBUINT16 *glyph_id_array = c->allocate_size<HBUINT16> (HBUINT16::static_size * num_codepoints);
        if (glyph_id_array == nullptr)
          return_trace (false);
        // From the cmap spec:
        //
        // id_range_offset[i]/2
        // + (cp - segments[i].start_code)
        // + (id_range_offset + i)
        // =
        // glyph_id_array + (cp - segments[i].start_code)
        //
        // So, solve for id_range_offset[i]:
        //
        // id_range_offset[i]
        // =
        // 2 * (glyph_id_array - id_range_offset - i)
        id_range_offset[i].set (2 * (
            glyph_id_array - id_range_offset - i));
        for (unsigned int j = 0; j < num_codepoints; j++)
        {
          hb_codepoint_t cp = segments[i].start_code + j;
          hb_codepoint_t new_gid;
          if (unlikely (!plan->new_gid_for_codepoint (cp, &new_gid)))
            return_trace (false);
          glyph_id_array[j].set (new_gid);
        }
      }
    }

    return_trace (true);
  }

  static inline size_t get_sub_table_size (const hb_vector_t<segment_plan> &segments)
  {
    size_t segment_size = 0;
    for (unsigned int i = 0; i < segments.len; i++)
    {
      // Parallel array entries
      segment_size +=
            2  // end count
          + 2  // start count
          + 2  // delta
          + 2; // range offset

      if (!segments[i].use_delta)
        // Add bytes for the glyph index array entries for this segment.
        segment_size += (segments[i].end_code - segments[i].start_code + 1) * 2;
    }

    return min_size
        + 2 // Padding
        + segment_size;
  }

  static inline bool create_sub_table_plan (const hb_subset_plan_t *plan,
                                            hb_vector_t<segment_plan> *segments)
  {
    segment_plan *segment = nullptr;
    hb_codepoint_t last_gid = 0;

    hb_codepoint_t cp = HB_SET_VALUE_INVALID;
    while (plan->unicodes->next (&cp)) {
      hb_codepoint_t new_gid;
      if (unlikely (!plan->new_gid_for_codepoint (cp, &new_gid)))
      {
	DEBUG_MSG(SUBSET, nullptr, "Unable to find new gid for %04x", cp);
	return false;
      }

      if (cp > 0xFFFF) {
        // We are now outside of unicode BMP, stop adding to this cmap.
        break;
      }

      if (!segment
          || cp != segment->end_code + 1u)
      {
        segment = segments->push ();
        segment->start_code.set (cp);
        segment->end_code.set (cp);
        segment->use_delta = true;
      } else {
        segment->end_code.set (cp);
        if (last_gid + 1u != new_gid)
          // gid's are not consecutive in this segment so delta
          // cannot be used.
          segment->use_delta = false;
      }

      last_gid = new_gid;
    }

    // There must be a final entry with end_code == 0xFFFF. Check if we need to add one.
    if (segment == nullptr || segment->end_code != 0xFFFF)
    {
      segment = segments->push ();
      segment->start_code.set (0xFFFF);
      segment->end_code.set (0xFFFF);
      segment->use_delta = true;
    }

    return true;
  }

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

    static inline void get_all_codepoints_func (const void *obj, hb_set_t *out)
    {
      const accelerator_t *thiz = (const accelerator_t *) obj;
      for (unsigned int i = 0; i < thiz->segCount; i++)
      {
	if (thiz->startCount[i] != 0xFFFFu
	    || thiz->endCount[i] != 0xFFFFu) // Skip the last segment (0xFFFF)
	  hb_set_add_range (out, thiz->startCount[i], thiz->endCount[i]);
      }
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
  HBUINT16	language;	/* Ignore. */
  HBUINT16	segCountX2;	/* 2 x segCount. */
  HBUINT16	searchRange;	/* 2 * (2**floor(log2(segCount))) */
  HBUINT16	entrySelector;	/* log2(searchRange/2) */
  HBUINT16	rangeShift;	/* 2 x segCount - searchRange */

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
  template<typename U>
  friend struct CmapSubtableLongSegmented;
  friend struct cmap;

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
  UINT		length;		/* Byte length of this subtable. */
  UINT		language;	/* Ignore. */
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
  friend struct cmap;

  inline bool get_glyph (hb_codepoint_t codepoint, hb_codepoint_t *glyph) const
  {
    int i = groups.bsearch (codepoint);
    if (i == -1)
      return false;
    *glyph = T::group_get_glyph (groups[i], codepoint);
    return true;
  }

  inline void get_all_codepoints (hb_set_t *out) const
  {
    for (unsigned int i = 0; i < this->groups.len; i++) {
      hb_set_add_range (out,
			MIN ((unsigned int) this->groups[i].startCharCode,
			     (unsigned int) HB_MAX_UNICODE_CODEPOINT_VALUE),
			MIN ((unsigned int) this->groups[i].endCharCode,
			     (unsigned int) HB_MAX_UNICODE_CODEPOINT_VALUE));
    }
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) && groups.sanitize (c));
  }

  inline bool serialize (hb_serialize_context_t *c,
                         const hb_vector_t<CmapSubtableLongGroup> &group_data)
  {
    TRACE_SERIALIZE (this);
    if (unlikely (!c->extend_min (*this))) return_trace (false);
    Supplier<CmapSubtableLongGroup> supplier (group_data.arrayZ, group_data.len);
    if (unlikely (!groups.serialize (c, supplier, group_data.len))) return_trace (false);
    return true;
  }

  protected:
  HBUINT16	format;		/* Subtable format; set to 12. */
  HBUINT16	reserved;	/* Reserved; set to 0. */
  HBUINT32	length;		/* Byte length of this subtable. */
  HBUINT32	language;	/* Ignore. */
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


  bool serialize (hb_serialize_context_t *c,
                  const hb_vector_t<CmapSubtableLongGroup> &groups)
  {
    if (unlikely (!c->extend_min (*this))) return false;

    this->format.set (12);
    this->reserved.set (0);
    this->length.set (get_sub_table_size (groups));

    return CmapSubtableLongSegmented<CmapSubtableFormat12>::serialize (c, groups);
  }

  static inline size_t get_sub_table_size (const hb_vector_t<CmapSubtableLongGroup> &groups)
  {
    return 16 + 12 * groups.len;
  }

  static inline bool create_sub_table_plan (const hb_subset_plan_t *plan,
                                            hb_vector_t<CmapSubtableLongGroup> *groups)
  {
    CmapSubtableLongGroup *group = nullptr;

    hb_codepoint_t cp = HB_SET_VALUE_INVALID;
    while (plan->unicodes->next (&cp)) {
      hb_codepoint_t new_gid;
      if (unlikely (!plan->new_gid_for_codepoint (cp, &new_gid)))
      {
	DEBUG_MSG(SUBSET, nullptr, "Unable to find new gid for %04x", cp);
	return false;
      }

      if (!group || !_is_gid_consecutive (group, cp, new_gid))
      {
        group = groups->push ();
        group->startCharCode.set (cp);
        group->endCharCode.set (cp);
        group->glyphID.set (new_gid);
      } else
      {
        group->endCharCode.set (cp);
      }
    }

    DEBUG_MSG(SUBSET, nullptr, "cmap");
    for (unsigned int i = 0; i < groups->len; i++) {
      CmapSubtableLongGroup& group = (*groups)[i];
      DEBUG_MSG(SUBSET, nullptr, "  %d: U+%04X-U+%04X, gid %d-%d", i, (uint32_t) group.startCharCode, (uint32_t) group.endCharCode, (uint32_t) group.glyphID, (uint32_t) group.glyphID + ((uint32_t) group.endCharCode - (uint32_t) group.startCharCode));
    }

    return true;
  }

 private:
  static inline bool _is_gid_consecutive (CmapSubtableLongGroup *group,
					  hb_codepoint_t cp,
					  hb_codepoint_t new_gid)
  {
    return (cp - 1 == group->endCharCode) &&
	new_gid == group->glyphID + (cp - group->startCharCode);
  }

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

  HBUINT24	startUnicodeValue;	/* First value in this range. */
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

  HBUINT24	unicodeValue;	/* Base Unicode value of the UVS */
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

  HBUINT24	varSelector;	/* Variation selector. */
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
  HBUINT32	length;		/* Byte length of this subtable. */
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
    case  0: return u.format0 .get_glyph (codepoint, glyph);
    case  4: return u.format4 .get_glyph (codepoint, glyph);
    case  6: return u.format6 .get_glyph (codepoint, glyph);
    case 10: return u.format10.get_glyph (codepoint, glyph);
    case 12: return u.format12.get_glyph (codepoint, glyph);
    case 13: return u.format13.get_glyph (codepoint, glyph);
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

  struct subset_plan {
    subset_plan(void)
    {
      format4_segments.init();
      format12_groups.init();
    }

    ~subset_plan(void)
    {
      format4_segments.fini();
      format12_groups.fini();
    }

    inline size_t final_size() const
    {
      return 4 // header
          +  8 * 3 // 3 EncodingRecord
          +  CmapSubtableFormat4::get_sub_table_size (this->format4_segments)
          +  CmapSubtableFormat12::get_sub_table_size (this->format12_groups);
    }

    // Format 4
    hb_vector_t<CmapSubtableFormat4::segment_plan> format4_segments;
    // Format 12
    hb_vector_t<CmapSubtableLongGroup> format12_groups;
  };

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this) &&
		  likely (version == 0) &&
		  encodingRecord.sanitize (c, this));
  }

  inline bool _create_plan (const hb_subset_plan_t *plan,
                            subset_plan *cmap_plan) const
  {
    if (unlikely( !CmapSubtableFormat4::create_sub_table_plan (plan, &cmap_plan->format4_segments)))
      return false;

    return CmapSubtableFormat12::create_sub_table_plan (plan, &cmap_plan->format12_groups);
  }

  inline bool _subset (const hb_subset_plan_t *plan,
                       const subset_plan &cmap_subset_plan,
		       size_t dest_sz,
		       void *dest) const
  {
    hb_serialize_context_t c (dest, dest_sz);

    cmap *table = c.start_serialize<cmap> ();
    if (unlikely (!c.extend_min (*table)))
    {
      return false;
    }

    table->version.set (0);

    if (unlikely (!table->encodingRecord.serialize (&c, /* numTables */ 3)))
      return false;

    // TODO(grieger): Convert the below to a for loop

    // Format 4, Plat 0 Encoding Record
    EncodingRecord &format4_plat0_rec = table->encodingRecord[0];
    format4_plat0_rec.platformID.set (0); // Unicode
    format4_plat0_rec.encodingID.set (3);

    // Format 4, Plat 3 Encoding Record
    EncodingRecord &format4_plat3_rec = table->encodingRecord[1];
    format4_plat3_rec.platformID.set (3); // Windows
    format4_plat3_rec.encodingID.set (1); // Unicode BMP

    // Format 12 Encoding Record
    EncodingRecord &format12_rec = table->encodingRecord[2];
    format12_rec.platformID.set (3); // Windows
    format12_rec.encodingID.set (10); // Unicode UCS-4

    // Write out format 4 sub table
    {
      CmapSubtable &subtable = format4_plat0_rec.subtable.serialize (&c, table);
      format4_plat3_rec.subtable.set (format4_plat0_rec.subtable);
      subtable.u.format.set (4);

      CmapSubtableFormat4 &format4 = subtable.u.format4;
      if (unlikely (!format4.serialize (&c, plan, cmap_subset_plan.format4_segments)))
        return false;
    }

    // Write out format 12 sub table.
    {
      CmapSubtable &subtable = format12_rec.subtable.serialize (&c, table);
      subtable.u.format.set (12);

      CmapSubtableFormat12 &format12 = subtable.u.format12;
      if (unlikely (!format12.serialize (&c, cmap_subset_plan.format12_groups)))
        return false;
    }

    c.end_serialize ();

    return true;
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    subset_plan cmap_subset_plan;

    if (unlikely (!_create_plan (plan, &cmap_subset_plan)))
    {
      DEBUG_MSG(SUBSET, nullptr, "Failed to generate a cmap subsetting plan.");
      return false;
    }

    // We now know how big our blob needs to be
    size_t dest_sz = cmap_subset_plan.final_size();
    void *dest = malloc (dest_sz);
    if (unlikely (!dest)) {
      DEBUG_MSG(SUBSET, nullptr, "Unable to alloc %lu for cmap subset output", (unsigned long) dest_sz);
      return false;
    }

    if (unlikely (!_subset (plan, cmap_subset_plan, dest_sz, dest)))
    {
      DEBUG_MSG(SUBSET, nullptr, "Failed to perform subsetting of cmap.");
      free (dest);
      return false;
    }

    // all done, write the blob into dest
    hb_blob_t *cmap_prime = hb_blob_create ((const char *)dest,
                                            dest_sz,
                                            HB_MEMORY_MODE_READONLY,
                                            dest,
                                            free);
    bool result =  plan->add_table (HB_OT_TAG_cmap, cmap_prime);
    hb_blob_destroy (cmap_prime);
    return result;
  }

  struct accelerator_t
  {
    inline void init (hb_face_t *face)
    {
      this->blob = hb_sanitize_context_t().reference_table<cmap> (face);
      const cmap *table = this->blob->as<cmap> ();
      const CmapSubtable *subtable = nullptr;
      const CmapSubtableFormat14 *subtable_uvs = nullptr;

      bool symbol = false;
      /* 32-bit subtables. */
      if (!subtable) subtable = table->find_subtable (3, 10);
      if (!subtable) subtable = table->find_subtable (0, 6);
      if (!subtable) subtable = table->find_subtable (0, 4);
      /* 16-bit subtables. */
      if (!subtable) subtable = table->find_subtable (3, 1);
      if (!subtable) subtable = table->find_subtable (0, 3);
      if (!subtable) subtable = table->find_subtable (0, 2);
      if (!subtable) subtable = table->find_subtable (0, 1);
      if (!subtable) subtable = table->find_subtable (0, 0);
      if (!subtable)
      {
	subtable = table->find_subtable (3, 0);
	if (subtable) symbol = true;
      }
      /* Meh. */
      if (!subtable) subtable = &Null(CmapSubtable);

      /* UVS subtable. */
      if (!subtable_uvs)
      {
	const CmapSubtable *st = table->find_subtable (0, 5);
	if (st && st->u.format == 14)
	  subtable_uvs = &st->u.format14;
      }
      /* Meh. */
      if (!subtable_uvs) subtable_uvs = &Null(CmapSubtableFormat14);

      this->uvs_table = subtable_uvs;

      this->get_glyph_data = subtable;
      if (unlikely (symbol))
      {
	this->get_glyph_func = get_glyph_from_symbol<CmapSubtable>;
	this->get_all_codepoints_func = null_get_all_codepoints_func;
      } else {
	switch (subtable->u.format) {
	/* Accelerate format 4 and format 12. */
	default:
	  this->get_glyph_func = get_glyph_from<CmapSubtable>;
	  this->get_all_codepoints_func = null_get_all_codepoints_func;
	  break;
	case 12:
	  this->get_glyph_func = get_glyph_from<CmapSubtableFormat12>;
	  this->get_all_codepoints_func = get_all_codepoints_from<CmapSubtableFormat12>;
	  break;
	case  4:
	  {
	    this->format4_accel.init (&subtable->u.format4);
	    this->get_glyph_data = &this->format4_accel;
	    this->get_glyph_func = this->format4_accel.get_glyph_func;
	    this->get_all_codepoints_func = this->format4_accel.get_all_codepoints_func;
	  }
	  break;
	}
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
	case GLYPH_VARIANT_NOT_FOUND:	return false;
	case GLYPH_VARIANT_FOUND:	return true;
	case GLYPH_VARIANT_USE_DEFAULT:	break;
      }

      return get_nominal_glyph (unicode, glyph);
    }

    inline void get_all_codepoints (hb_set_t *out) const
    {
      this->get_all_codepoints_func (get_glyph_data, out);
    }

    protected:
    typedef bool (*hb_cmap_get_glyph_func_t) (const void *obj,
					      hb_codepoint_t codepoint,
					      hb_codepoint_t *glyph);
    typedef void (*hb_cmap_get_all_codepoints_func_t) (const void *obj,
						       hb_set_t *out);

    static inline void null_get_all_codepoints_func (const void *obj, hb_set_t *out)
    {
      // NOOP
    }

    template <typename Type>
    static inline bool get_glyph_from (const void *obj,
				       hb_codepoint_t codepoint,
				       hb_codepoint_t *glyph)
    {
      const Type *typed_obj = (const Type *) obj;
      return typed_obj->get_glyph (codepoint, glyph);
    }

    template <typename Type>
    static inline void get_all_codepoints_from (const void *obj,
						hb_set_t *out)
    {
      const Type *typed_obj = (const Type *) obj;
      typed_obj->get_all_codepoints (out);
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
	 * https://docs.microsoft.com/en-us/typography/opentype/spec/recom
	 * under "Non-Standard (Symbol) Fonts". */
	return typed_obj->get_glyph (0xF000u + codepoint, glyph);
      }

      return false;
    }

    private:
    hb_cmap_get_glyph_func_t get_glyph_func;
    const void *get_glyph_data;
    hb_cmap_get_all_codepoints_func_t get_all_codepoints_func;

    CmapSubtableFormat4::accelerator_t format4_accel;

    const CmapSubtableFormat14 *uvs_table;
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
