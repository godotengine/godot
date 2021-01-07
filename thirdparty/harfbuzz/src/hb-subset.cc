/*
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
 * Google Author(s): Garret Rieger, Rod Sheeter, Behdad Esfahbod
 */

#include "hb.hh"
#include "hb-open-type.hh"

#include "hb-subset.hh"

#include "hb-open-file.hh"
#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-hdmx-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-maxp-table.hh"
#include "hb-ot-color-sbix-table.hh"
#include "hb-ot-color-colr-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-post-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"
#include "hb-ot-vorg-table.hh"
#include "hb-ot-name-table.hh"
#include "hb-ot-color-cbdt-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-layout-gpos-table.hh"
#include "hb-ot-var-gvar-table.hh"
#include "hb-ot-var-hvar-table.hh"


static unsigned
_plan_estimate_subset_table_size (hb_subset_plan_t *plan, unsigned table_len)
{
  unsigned src_glyphs = plan->source->get_num_glyphs ();
  unsigned dst_glyphs = plan->glyphset ()->get_population ();

  if (unlikely (!src_glyphs))
    return 512 + table_len;

  return 512 + (unsigned) (table_len * sqrt ((double) dst_glyphs / src_glyphs));
}

template<typename TableType>
static bool
_subset (hb_subset_plan_t *plan)
{
  bool result = false;
  hb_blob_t *source_blob = hb_sanitize_context_t ().reference_table<TableType> (plan->source);
  const TableType *table = source_blob->as<TableType> ();

  hb_tag_t tag = TableType::tableTag;
  if (source_blob->data)
  {
    hb_vector_t<char> buf;
    /* TODO Not all tables are glyph-related.  'name' table size for example should not be
     * affected by number of glyphs.  Accommodate that. */
    unsigned buf_size = _plan_estimate_subset_table_size (plan, source_blob->length);
    DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c initial estimated table size: %u bytes.", HB_UNTAG (tag), buf_size);
    if (unlikely (!buf.alloc (buf_size)))
    {
      DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c failed to allocate %u bytes.", HB_UNTAG (tag), buf_size);
      hb_blob_destroy (source_blob);
      return false;
    }
  retry:
    hb_serialize_context_t serializer ((void *) buf, buf_size);
    serializer.start_serialize<TableType> ();
    hb_subset_context_t c (source_blob, plan, &serializer, tag);
    bool needed = table->subset (&c);
    if (serializer.ran_out_of_room)
    {
      buf_size += (buf_size >> 1) + 32;
      DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c ran out of room; reallocating to %u bytes.", HB_UNTAG (tag), buf_size);
      if (unlikely (!buf.alloc (buf_size)))
      {
	DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c failed to reallocate %u bytes.", HB_UNTAG (tag), buf_size);
	hb_blob_destroy (source_blob);
	return false;
      }
      goto retry;
    }
    serializer.end_serialize ();

    result = !serializer.in_error ();

    if (result)
    {
      if (needed)
      {
	hb_blob_t *dest_blob = serializer.copy_blob ();
	DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c final subset table size: %u bytes.", HB_UNTAG (tag), dest_blob->length);
	result = c.plan->add_table (tag, dest_blob);
	hb_blob_destroy (dest_blob);
      }
      else
      {
	DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c::subset table subsetted to empty.", HB_UNTAG (tag));
      }
    }
  }
  else
    DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c::subset sanitize failed on source table.", HB_UNTAG (tag));

  hb_blob_destroy (source_blob);
  DEBUG_MSG (SUBSET, nullptr, "OT::%c%c%c%c::subset %s", HB_UNTAG (tag), result ? "success" : "FAILED!");
  return result;
}

static bool
_is_table_present (hb_face_t *source, hb_tag_t tag)
{
  hb_tag_t table_tags[32];
  unsigned offset = 0, num_tables = ARRAY_LENGTH (table_tags);
  while ((hb_face_get_table_tags (source, offset, &num_tables, table_tags), num_tables))
  {
    for (unsigned i = 0; i < num_tables; ++i)
      if (table_tags[i] == tag)
	return true;
    offset += num_tables;
  }
  return false;
}

static bool
_should_drop_table (hb_subset_plan_t *plan, hb_tag_t tag)
{
  if (plan->drop_tables->has (tag))
    return true;

  switch (tag)
  {
  case HB_TAG ('c','v','a','r'): /* hint table, fallthrough */
  case HB_TAG ('c','v','t',' '): /* hint table, fallthrough */
  case HB_TAG ('f','p','g','m'): /* hint table, fallthrough */
  case HB_TAG ('p','r','e','p'): /* hint table, fallthrough */
  case HB_TAG ('h','d','m','x'): /* hint table, fallthrough */
  case HB_TAG ('V','D','M','X'): /* hint table, fallthrough */
    return plan->drop_hints;

#ifdef HB_NO_SUBSET_LAYOUT
    // Drop Layout Tables if requested.
  case HB_OT_TAG_GDEF:
  case HB_OT_TAG_GPOS:
  case HB_OT_TAG_GSUB:
  case HB_TAG ('m','o','r','x'):
  case HB_TAG ('m','o','r','t'):
  case HB_TAG ('k','e','r','x'):
  case HB_TAG ('k','e','r','n'):
    return true;
#endif

  default:
    return false;
  }
}

static bool
_subset_table (hb_subset_plan_t *plan, hb_tag_t tag)
{
  DEBUG_MSG (SUBSET, nullptr, "subset %c%c%c%c", HB_UNTAG (tag));
  switch (tag)
  {
  case HB_OT_TAG_glyf: return _subset<const OT::glyf> (plan);
  case HB_OT_TAG_hdmx: return _subset<const OT::hdmx> (plan);
  case HB_OT_TAG_name: return _subset<const OT::name> (plan);
  case HB_OT_TAG_head:
    if (_is_table_present (plan->source, HB_OT_TAG_glyf) && !_should_drop_table (plan, HB_OT_TAG_glyf))
      return true; /* skip head, handled by glyf */
    return _subset<const OT::head> (plan);
  case HB_OT_TAG_hhea: return true; /* skip hhea, handled by hmtx */
  case HB_OT_TAG_hmtx: return _subset<const OT::hmtx> (plan);
  case HB_OT_TAG_vhea: return true; /* skip vhea, handled by vmtx */
  case HB_OT_TAG_vmtx: return _subset<const OT::vmtx> (plan);
  case HB_OT_TAG_maxp: return _subset<const OT::maxp> (plan);
  case HB_OT_TAG_sbix: return _subset<const OT::sbix> (plan);
  case HB_OT_TAG_loca: return true; /* skip loca, handled by glyf */
  case HB_OT_TAG_cmap: return _subset<const OT::cmap> (plan);
  case HB_OT_TAG_OS2 : return _subset<const OT::OS2 > (plan);
  case HB_OT_TAG_post: return _subset<const OT::post> (plan);
  case HB_OT_TAG_COLR: return _subset<const OT::COLR> (plan);
  case HB_OT_TAG_CBLC: return _subset<const OT::CBLC> (plan);
  case HB_OT_TAG_CBDT: return true; /* skip CBDT, handled by CBLC */

#ifndef HB_NO_SUBSET_CFF
  case HB_OT_TAG_cff1: return _subset<const OT::cff1> (plan);
  case HB_OT_TAG_cff2: return _subset<const OT::cff2> (plan);
  case HB_OT_TAG_VORG: return _subset<const OT::VORG> (plan);
#endif

#ifndef HB_NO_SUBSET_LAYOUT
  case HB_OT_TAG_GDEF: return _subset<const OT::GDEF> (plan);
  case HB_OT_TAG_GSUB: return _subset<const OT::GSUB> (plan);
  case HB_OT_TAG_GPOS: return _subset<const OT::GPOS> (plan);
  case HB_OT_TAG_gvar: return _subset<const OT::gvar> (plan);
  case HB_OT_TAG_HVAR: return _subset<const OT::HVAR> (plan);
  case HB_OT_TAG_VVAR: return _subset<const OT::VVAR> (plan);
#endif

  default:
    hb_blob_t *source_table = hb_face_reference_table (plan->source, tag);
    bool result = plan->add_table (tag, source_table);
    hb_blob_destroy (source_table);
    return result;
  }
}

/**
 * hb_subset:
 * @source: font face data to be subset.
 * @input: input to use for the subsetting.
 *
 * Subsets a font according to provided input.
 **/
hb_face_t *
hb_subset (hb_face_t *source, hb_subset_input_t *input)
{
  if (unlikely (!input || !source)) return hb_face_get_empty ();

  hb_subset_plan_t *plan = hb_subset_plan_create (source, input);
  if (unlikely (plan->in_error ()))
    return hb_face_get_empty ();

  hb_set_t tags_set;
  bool success = true;
  hb_tag_t table_tags[32];
  unsigned offset = 0, num_tables = ARRAY_LENGTH (table_tags);
  while ((hb_face_get_table_tags (source, offset, &num_tables, table_tags), num_tables))
  {
    for (unsigned i = 0; i < num_tables; ++i)
    {
      hb_tag_t tag = table_tags[i];
      if (_should_drop_table (plan, tag) && !tags_set.has (tag)) continue;
      tags_set.add (tag);
      success = _subset_table (plan, tag);
      if (unlikely (!success)) goto end;
    }
    offset += num_tables;
  }
end:

  hb_face_t *result = success ? hb_face_reference (plan->dest) : hb_face_get_empty ();

  hb_subset_plan_destroy (plan);
  return result;
}
