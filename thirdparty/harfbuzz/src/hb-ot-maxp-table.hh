/*
 * Copyright © 2011,2012  Google, Inc.
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

#ifndef HB_OT_MAXP_TABLE_HH
#define HB_OT_MAXP_TABLE_HH

#include "hb-open-type-private.hh"
#include "hb-subset-plan.hh"

namespace OT {


/*
 * maxp -- Maximum Profile
 * https://docs.microsoft.com/en-us/typography/opentype/spec/maxp
 */

#define HB_OT_TAG_maxp HB_TAG('m','a','x','p')

struct maxpV1Tail
{
  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  HBUINT16 maxPoints;		  /* Maximum points in a non-composite glyph. */
  HBUINT16 maxContours;		  /* Maximum contours in a non-composite glyph. */
  HBUINT16 maxCompositePoints;	  /* Maximum points in a composite glyph. */
  HBUINT16 maxCompositeContours;  /* Maximum contours in a composite glyph. */
  HBUINT16 maxZones;		  /* 1 if instructions do not use the twilight zone (Z0),
				   * or 2 if instructions do use Z0; should be set to 2 in
				   * most cases. */
  HBUINT16 maxTwilightPoints;	  /* Maximum points used in Z0. */
  HBUINT16 maxStorage;		  /* Number of Storage Area locations. */
  HBUINT16 maxFunctionDefs;	  /* Number of FDEFs, equal to the highest function number + 1. */
  HBUINT16 maxInstructionDefs;	  /* Number of IDEFs. */
  HBUINT16 maxStackElements;	  /* Maximum stack depth. (This includes Font and CVT
				   * Programs, as well as the instructions for each glyph.) */
  HBUINT16 maxSizeOfInstructions; /* Maximum byte count for glyph instructions. */
  HBUINT16 maxComponentElements;  /* Maximum number of components referenced at
				   * "top level" for any composite glyph. */
  HBUINT16 maxComponentDepth;	  /* Maximum levels of recursion; 1 for simple components. */
 public:
  DEFINE_SIZE_STATIC (26);
};


struct maxp
{
  static const hb_tag_t tableTag = HB_OT_TAG_maxp;

  inline unsigned int get_num_glyphs (void) const
  {
    return numGlyphs;
  }

  inline void set_num_glyphs (unsigned int count)
  {
    numGlyphs.set (count);
  }

  inline bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    if (unlikely (!c->check_struct (this)))
      return_trace (false);

    if (version.major == 1)
    {
      const maxpV1Tail &v1 = StructAfter<maxpV1Tail> (*this);
      return v1.sanitize (c);
    }
    return_trace (likely (version.major == 0 && version.minor == 0x5000u));
  }

  inline bool subset (hb_subset_plan_t *plan) const
  {
    hb_blob_t *maxp_blob = hb_sanitize_context_t().reference_table<maxp> (plan->source);
    hb_blob_t *maxp_prime_blob = hb_blob_copy_writable_or_fail (maxp_blob);
    hb_blob_destroy (maxp_blob);

    if (unlikely (!maxp_prime_blob)) {
      return false;
    }
    maxp *maxp_prime = (maxp *) hb_blob_get_data (maxp_prime_blob, nullptr);

    maxp_prime->set_num_glyphs (plan->glyphs.len);
    if (plan->drop_hints)
      drop_hint_fields (plan, maxp_prime);

    bool result = plan->add_table (HB_OT_TAG_maxp, maxp_prime_blob);
    hb_blob_destroy (maxp_prime_blob);
    return result;
  }

  static inline void drop_hint_fields (hb_subset_plan_t *plan, maxp *maxp_prime)
  {
    if (maxp_prime->version.major == 1)
    {
      maxpV1Tail &v1 = StructAfter<maxpV1Tail> (*maxp_prime);
      v1.maxZones.set (1);
      v1.maxTwilightPoints.set (0);
      v1.maxStorage.set (0);
      v1.maxFunctionDefs.set (0);
      v1.maxInstructionDefs.set (0);
      v1.maxStackElements.set (0);
      v1.maxSizeOfInstructions.set (0);
    }
  }

  protected:
  FixedVersion<>version;		/* Version of the maxp table (0.5 or 1.0),
					 * 0x00005000u or 0x00010000u. */
  HBUINT16	numGlyphs;		/* The number of glyphs in the font. */
/*maxpV1Tail v1Tail[VAR]; */
  public:
  DEFINE_SIZE_STATIC (6);
};


} /* namespace OT */


#endif /* HB_OT_MAXP_TABLE_HH */
