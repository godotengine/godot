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

#ifndef HB_OT_VAR_MVAR_TABLE_HH
#define HB_OT_VAR_MVAR_TABLE_HH

#include "hb-ot-var-common.hh"


namespace OT {


struct VariationValueRecord
{
  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (c->check_struct (this));
  }

  bool subset (hb_subset_context_t *c,
               const hb_map_t& varidx_map) const
  {
    TRACE_SUBSET (this);
    auto *out = c->serializer->embed (*this);
    if (unlikely (!out)) return_trace (false);

    hb_codepoint_t *new_idx;
    return_trace (c->serializer->check_assign (out->varIdx,
                                               (varidx_map.has (varIdx, &new_idx)) ? *new_idx : HB_OT_LAYOUT_NO_VARIATIONS_INDEX,
                                               HB_SERIALIZE_ERROR_INT_OVERFLOW));
  }

  public:
  Tag		valueTag;	/* Four-byte tag identifying a font-wide measure. */
  VarIdx	varIdx;		/* Outer/inner index into VariationStore item. */

  public:
  DEFINE_SIZE_STATIC (8);
};


/*
 * MVAR -- Metrics Variations
 * https://docs.microsoft.com/en-us/typography/opentype/spec/mvar
 */
#define HB_OT_TAG_MVAR HB_TAG('M','V','A','R')

struct MVAR
{
  static constexpr hb_tag_t tableTag = HB_OT_TAG_MVAR;

  bool sanitize (hb_sanitize_context_t *c) const
  {
    TRACE_SANITIZE (this);
    return_trace (version.sanitize (c) &&
		  likely (version.major == 1) &&
		  c->check_struct (this) &&
		  valueRecordSize >= VariationValueRecord::static_size &&
		  varStore.sanitize (c, this) &&
		  c->check_range (valuesZ.arrayZ,
				  valueRecordCount,
				  valueRecordSize));
  }

  bool subset (hb_subset_context_t *c) const
  {
    TRACE_SUBSET (this);
#ifdef HB_NO_VAR
    return_trace (false);
#endif

    if (c->plan->all_axes_pinned)
      return_trace (false);

    MVAR *out = c->serializer->start_embed (*this);
    if (unlikely (!c->serializer->extend_min (out))) return_trace (false);
    out->version = version;
    out->reserved = reserved;
    out->valueRecordSize = valueRecordSize;
    out->valueRecordCount = valueRecordCount;

    item_variations_t item_vars;
    const VariationStore& src_var_store = this+varStore;

    if (!item_vars.instantiate (src_var_store, c->plan))
      return_trace (false);

    /* serialize varstore */
    if (!out->varStore.serialize_serialize (c->serializer, item_vars.has_long_word (),
                                            c->plan->axis_tags,
                                            item_vars.get_region_list (),
                                            item_vars.get_vardata_encodings ()))
      return_trace (false);

    /* serialize value records array */
    unsigned value_rec_count = valueRecordCount;
    const VariationValueRecord *record = reinterpret_cast<const VariationValueRecord*> (valuesZ.arrayZ);
    for (unsigned i = 0; i < value_rec_count; i++)
    {
      if (!record->subset (c, item_vars.get_varidx_map ())) return_trace (false);
      record++;
    }
    return_trace (true);
  }

  float get_var (hb_tag_t tag,
		 const int *coords, unsigned int coord_count) const
  {
    const VariationValueRecord *record;
    record = (VariationValueRecord *) hb_bsearch (tag,
						  (const VariationValueRecord *)
						    (const HBUINT8 *) valuesZ,
						  valueRecordCount, valueRecordSize,
						  tag_compare);
    if (!record)
      return 0.;

    return (this+varStore).get_delta (record->varIdx, coords, coord_count);
  }

protected:
  static int tag_compare (const void *pa, const void *pb)
  {
    const hb_tag_t *a = (const hb_tag_t *) pa;
    const Tag *b = (const Tag *) pb;
    return b->cmp (*a);
  }

  protected:
  FixedVersion<>version;	/* Version of the metrics variation table
				 * initially set to 0x00010000u */
  HBUINT16	reserved;	/* Not used; set to 0. */
  HBUINT16	valueRecordSize;/* The size in bytes of each value record —
				 * must be greater than zero. */
  HBUINT16	valueRecordCount;/* The number of value records — may be zero. */
  Offset16To<VariationStore>
		varStore;	/* Offset to item variation store table. */
  UnsizedArrayOf<HBUINT8>
		valuesZ;	/* Array of value records. The records must be
				 * in binary order of their valueTag field. */

  public:
  DEFINE_SIZE_ARRAY (12, valuesZ);
};

} /* namespace OT */


#define HB_ADD_MVAR_VAR(tag, field) \
       c->serializer->check_assign (table->field, \
				    roundf (table->field + \
					    MVAR.get_var (tag, \
							  c->plan->normalized_coords.arrayZ, \
							  c->plan->normalized_coords.length)), \
				    HB_SERIALIZE_ERROR_INT_OVERFLOW)


#endif /* HB_OT_VAR_MVAR_TABLE_HH */
