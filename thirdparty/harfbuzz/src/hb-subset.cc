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

#include "hb-private.hh"
#include "hb-open-type-private.hh"

#include "hb-subset-glyf.hh"
#include "hb-subset-private.hh"
#include "hb-subset-plan.hh"

#include "hb-open-file-private.hh"
#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-hdmx-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-hhea-table.hh"
#include "hb-ot-hmtx-table.hh"
#include "hb-ot-maxp-table.hh"
#include "hb-ot-os2-table.hh"
#include "hb-ot-post-table.hh"


struct hb_subset_profile_t {
  hb_object_header_t header;
  ASSERT_POD ();
};

/**
 * hb_subset_profile_create:
 *
 * Return value: New profile with default settings.
 *
 * Since: 1.8.0
 **/
hb_subset_profile_t *
hb_subset_profile_create ()
{
  return hb_object_create<hb_subset_profile_t>();
}

/**
 * hb_subset_profile_destroy:
 *
 * Since: 1.8.0
 **/
void
hb_subset_profile_destroy (hb_subset_profile_t *profile)
{
  if (!hb_object_destroy (profile)) return;

  free (profile);
}

template<typename TableType>
static bool
_subset (hb_subset_plan_t *plan)
{
  hb_blob_t *source_blob = hb_sanitize_context_t ().reference_table<TableType> (plan->source);
  const TableType *table = source_blob->as<TableType> ();

  hb_tag_t tag = TableType::tableTag;
  hb_bool_t result = false;
  if (source_blob->data)
  {
    result = table->subset(plan);
  } else {
    DEBUG_MSG(SUBSET, nullptr, "OT::%c%c%c%c::subset sanitize failed on source table.", HB_UNTAG(tag));
  }

  hb_blob_destroy (source_blob);
  DEBUG_MSG(SUBSET, nullptr, "OT::%c%c%c%c::subset %s", HB_UNTAG(tag), result ? "success" : "FAILED!");
  return result;
}


/*
 * A face that has add_table().
 */

struct hb_subset_face_data_t
{
  struct table_entry_t
  {
    inline int cmp (const hb_tag_t *t) const
    {
      if (*t < tag) return -1;
      if (*t > tag) return -1;
      return 0;
    }

    hb_tag_t   tag;
    hb_blob_t *blob;
  };

  hb_vector_t<table_entry_t, 32> tables;
};

static hb_subset_face_data_t *
_hb_subset_face_data_create (void)
{
  hb_subset_face_data_t *data = (hb_subset_face_data_t *) calloc (1, sizeof (hb_subset_face_data_t));
  if (unlikely (!data))
    return nullptr;

  data->tables.init ();

  return data;
}

static void
_hb_subset_face_data_destroy (void *user_data)
{
  hb_subset_face_data_t *data = (hb_subset_face_data_t *) user_data;

  for (unsigned int i = 0; i < data->tables.len; i++)
    hb_blob_destroy (data->tables[i].blob);

  data->tables.fini ();

  free (data);
}

static hb_blob_t *
_hb_subset_face_data_reference_blob (hb_subset_face_data_t *data)
{

  unsigned int table_count = data->tables.len;
  unsigned int face_length = table_count * 16 + 12;

  for (unsigned int i = 0; i < table_count; i++)
    face_length += hb_ceil_to_4 (hb_blob_get_length (data->tables.arrayZ[i].blob));

  char *buf = (char *) malloc (face_length);
  if (unlikely (!buf))
    return nullptr;

  hb_serialize_context_t c (buf, face_length);
  OT::OpenTypeFontFile *f = c.start_serialize<OT::OpenTypeFontFile> ();

  bool is_cff = data->tables.lsearch (HB_TAG ('C','F','F',' ')) || data->tables.lsearch (HB_TAG ('C','F','F','2'));
  hb_tag_t sfnt_tag = is_cff ? OT::OpenTypeFontFile::CFFTag : OT::OpenTypeFontFile::TrueTypeTag;

  Supplier<hb_tag_t>    tags_supplier  (&data->tables[0].tag, table_count, sizeof (data->tables[0]));
  Supplier<hb_blob_t *> blobs_supplier (&data->tables[0].blob, table_count, sizeof (data->tables[0]));
  bool ret = f->serialize_single (&c,
				  sfnt_tag,
				  tags_supplier,
				  blobs_supplier,
				  table_count);

  c.end_serialize ();

  if (unlikely (!ret))
  {
    free (buf);
    return nullptr;
  }

  return hb_blob_create (buf, face_length, HB_MEMORY_MODE_WRITABLE, buf, free);
}

static hb_blob_t *
_hb_subset_face_reference_table (hb_face_t *face, hb_tag_t tag, void *user_data)
{
  hb_subset_face_data_t *data = (hb_subset_face_data_t *) user_data;

  if (!tag)
    return _hb_subset_face_data_reference_blob (data);

  hb_subset_face_data_t::table_entry_t *entry = data->tables.lsearch (tag);
  if (entry)
    return hb_blob_reference (entry->blob);

  return nullptr;
}

/* TODO: Move this to hb-face.h and rename to hb_face_builder_create()
 * with hb_face_builder_add_table(). */
hb_face_t *
hb_subset_face_create (void)
{
  hb_subset_face_data_t *data = _hb_subset_face_data_create ();
  if (unlikely (!data)) return hb_face_get_empty ();

  return hb_face_create_for_tables (_hb_subset_face_reference_table,
				    data,
				    _hb_subset_face_data_destroy);
}

hb_bool_t
hb_subset_face_add_table (hb_face_t *face, hb_tag_t tag, hb_blob_t *blob)
{
  if (unlikely (face->destroy != (hb_destroy_func_t) _hb_subset_face_data_destroy))
    return false;

  hb_subset_face_data_t *data = (hb_subset_face_data_t *) face->user_data;
  hb_subset_face_data_t::table_entry_t *entry = data->tables.push ();

  entry->tag = tag;
  entry->blob = hb_blob_reference (blob);

  return true;
}

static bool
_subset_table (hb_subset_plan_t *plan,
               hb_tag_t          tag)
{
  DEBUG_MSG(SUBSET, nullptr, "begin subset %c%c%c%c", HB_UNTAG(tag));
  bool result = true;
  switch (tag) {
    case HB_OT_TAG_glyf:
      result = _subset<const OT::glyf> (plan);
      break;
    case HB_OT_TAG_hdmx:
      result = _subset<const OT::hdmx> (plan);
      break;
    case HB_OT_TAG_head:
      // TODO that won't work well if there is no glyf
      DEBUG_MSG(SUBSET, nullptr, "skip head, handled by glyf");
      result = true;
      break;
    case HB_OT_TAG_hhea:
      DEBUG_MSG(SUBSET, nullptr, "skip hhea handled by hmtx");
      return true;
    case HB_OT_TAG_hmtx:
      result = _subset<const OT::hmtx> (plan);
      break;
    case HB_OT_TAG_vhea:
      DEBUG_MSG(SUBSET, nullptr, "skip vhea handled by vmtx");
      return true;
    case HB_OT_TAG_vmtx:
      result = _subset<const OT::vmtx> (plan);
      break;
    case HB_OT_TAG_maxp:
      result = _subset<const OT::maxp> (plan);
      break;
    case HB_OT_TAG_loca:
      DEBUG_MSG(SUBSET, nullptr, "skip loca handled by glyf");
      return true;
    case HB_OT_TAG_cmap:
      result = _subset<const OT::cmap> (plan);
      break;
    case HB_OT_TAG_os2:
      result = _subset<const OT::os2> (plan);
      break;
    case HB_OT_TAG_post:
      result = _subset<const OT::post> (plan);
      break;
    default:
      hb_blob_t *source_table = hb_face_reference_table(plan->source, tag);
      if (likely (source_table))
        result = plan->add_table(tag, source_table);
      else
        result = false;
      hb_blob_destroy (source_table);
      break;
  }
  DEBUG_MSG(SUBSET, nullptr, "subset %c%c%c%c %s", HB_UNTAG(tag), result ? "ok" : "FAILED");
  return result;
}

static bool
_should_drop_table(hb_subset_plan_t *plan, hb_tag_t tag)
{
  switch (tag) {
    case HB_TAG ('c', 'v', 'a', 'r'): /* hint table, fallthrough */
    case HB_TAG ('c', 'v', 't', ' '): /* hint table, fallthrough */
    case HB_TAG ('f', 'p', 'g', 'm'): /* hint table, fallthrough */
    case HB_TAG ('p', 'r', 'e', 'p'): /* hint table, fallthrough */
    case HB_TAG ('h', 'd', 'm', 'x'): /* hint table, fallthrough */
    case HB_TAG ('V', 'D', 'M', 'X'): /* hint table, fallthrough */
      return plan->drop_hints;
    // Drop Layout Tables if requested.
    case HB_TAG ('G', 'D', 'E', 'F'): /* temporary */
    case HB_TAG ('G', 'P', 'O', 'S'): /* temporary */
    case HB_TAG ('G', 'S', 'U', 'B'): /* temporary */
      return plan->drop_ot_layout;
    // Drop these tables below by default, list pulled
    // from fontTools:
    case HB_TAG ('B', 'A', 'S', 'E'):
    case HB_TAG ('J', 'S', 'T', 'F'):
    case HB_TAG ('D', 'S', 'I', 'G'):
    case HB_TAG ('E', 'B', 'D', 'T'):
    case HB_TAG ('E', 'B', 'L', 'C'):
    case HB_TAG ('E', 'B', 'S', 'C'):
    case HB_TAG ('S', 'V', 'G', ' '):
    case HB_TAG ('P', 'C', 'L', 'T'):
    case HB_TAG ('L', 'T', 'S', 'H'):
    // Graphite tables:
    case HB_TAG ('F', 'e', 'a', 't'):
    case HB_TAG ('G', 'l', 'a', 't'):
    case HB_TAG ('G', 'l', 'o', 'c'):
    case HB_TAG ('S', 'i', 'l', 'f'):
    case HB_TAG ('S', 'i', 'l', 'l'):
    // Colour
    case HB_TAG ('s', 'b', 'i', 'x'):
      return true;
    default:
      return false;
  }
}

/**
 * hb_subset:
 * @source: font face data to be subset.
 * @profile: profile to use for the subsetting.
 * @input: input to use for the subsetting.
 *
 * Subsets a font according to provided profile and input.
 **/
hb_face_t *
hb_subset (hb_face_t *source,
           hb_subset_profile_t *profile,
           hb_subset_input_t *input)
{
  if (unlikely (!profile || !input || !source)) return hb_face_get_empty();

  hb_subset_plan_t *plan = hb_subset_plan_create (source, profile, input);

  hb_tag_t table_tags[32];
  unsigned int offset = 0, count;
  bool success = true;
  do {
    count = ARRAY_LENGTH (table_tags);
    hb_face_get_table_tags (source, offset, &count, table_tags);
    for (unsigned int i = 0; i < count; i++)
    {
      hb_tag_t tag = table_tags[i];
      if (_should_drop_table(plan, tag))
      {
        DEBUG_MSG(SUBSET, nullptr, "drop %c%c%c%c", HB_UNTAG(tag));
        continue;
      }
      success = success && _subset_table (plan, tag);
    }
    offset += count;
  } while (count == ARRAY_LENGTH (table_tags));

  hb_face_t *result = success ? hb_face_reference(plan->dest) : hb_face_get_empty();
  hb_subset_plan_destroy (plan);
  return result;
}

/**
 * hb_subset_get_all_codepoints:
 * @source: font face data to load.
 * @out: set to add the all codepoints covered by font face, source.
 */
void
hb_subset_get_all_codepoints (hb_face_t *source, hb_set_t *out)
{
  OT::cmap::accelerator_t cmap;
  cmap.init (source);
  cmap.get_all_codepoints (out);
  cmap.fini();
}
