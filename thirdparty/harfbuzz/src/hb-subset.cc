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
#include "hb-open-file.hh"

#include "hb-subset.hh"
#include "hb-subset-table.hh"
#include "hb-subset-accelerator.hh"

#include "hb-ot-cmap-table.hh"
#include "hb-ot-var-cvar-table.hh"
#include "hb-ot-head-table.hh"
#include "hb-ot-stat-table.hh"
#include "hb-ot-post-table-v2subset.hh"


/**
 * SECTION:hb-subset
 * @title: hb-subset
 * @short_description: Subsets font files.
 * @include: hb-subset.h
 *
 * Subsetting reduces the codepoint coverage of font files and removes all data
 * that is no longer needed. A subset input describes the desired subset. The input is
 * provided along with a font to the subsetting operation. Output is a new font file
 * containing only the data specified in the input.
 *
 * Currently most outline and bitmap tables are supported: glyf, CFF, CFF2, sbix,
 * COLR, and CBDT/CBLC. This also includes fonts with variable outlines via OpenType
 * variations. Notably EBDT/EBLC and SVG are not supported. Layout subsetting is supported
 * only for OpenType Layout tables (GSUB, GPOS, GDEF). Notably subsetting of graphite or AAT tables
 * is not yet supported.
 *
 * Fonts with graphite or AAT tables may still be subsetted but will likely need to use the
 * retain glyph ids option and configure the subset to pass through the layout tables untouched.
 */


hb_user_data_key_t _hb_subset_accelerator_user_data_key = {};


/*
 * The list of tables in the open type spec. Used to check for tables that may need handling
 * if we are unable to list the tables in a face.
 */
static hb_tag_t known_tables[] {
  HB_TAG('a','v','a','r'),
  HB_TAG('B','A','S','E'),
  HB_TAG('C','B','D','T'),
  HB_TAG('C','B','L','C'),
  HB_TAG('C','F','F',' '),
  HB_TAG('C','F','F','2'),
  HB_TAG('c','m','a','p'),
  HB_TAG('C','O','L','R'),
  HB_TAG('C','P','A','L'),
  HB_TAG('c','v','a','r'),
  HB_TAG('c','v','t',' '),
  HB_TAG('D','S','I','G'),
  HB_TAG('E','B','D','T'),
  HB_TAG('E','B','L','C'),
  HB_TAG('E','B','S','C'),
  HB_TAG('f','p','g','m'),
  HB_TAG('f','v','a','r'),
  HB_TAG('g','a','s','p'),
  HB_TAG('G','D','E','F'),
  HB_TAG('g','l','y','f'),
  HB_TAG('G','P','O','S'),
  HB_TAG('G','S','U','B'),
  HB_TAG('g','v','a','r'),
  HB_TAG('h','d','m','x'),
  HB_TAG('h','e','a','d'),
  HB_TAG('h','h','e','a'),
  HB_TAG('h','m','t','x'),
  HB_TAG('H','V','A','R'),
  HB_TAG('J','S','T','F'),
  HB_TAG('k','e','r','n'),
  HB_TAG('l','o','c','a'),
  HB_TAG('L','T','S','H'),
  HB_TAG('M','A','T','H'),
  HB_TAG('m','a','x','p'),
  HB_TAG('M','E','R','G'),
  HB_TAG('m','e','t','a'),
  HB_TAG('M','V','A','R'),
  HB_TAG('P','C','L','T'),
  HB_TAG('p','o','s','t'),
  HB_TAG('p','r','e','p'),
  HB_TAG('s','b','i','x'),
  HB_TAG('S','T','A','T'),
  HB_TAG('S','V','G',' '),
  HB_TAG('V','D','M','X'),
  HB_TAG('v','h','e','a'),
  HB_TAG('v','m','t','x'),
  HB_TAG('V','O','R','G'),
  HB_TAG('V','V','A','R'),
  HB_TAG('n','a','m','e'),
  HB_TAG('O','S','/','2')
};

static bool _table_is_empty (const hb_face_t *face, hb_tag_t tag)
{
  hb_blob_t* blob = hb_face_reference_table (face, tag);
  bool result = (blob == hb_blob_get_empty ());
  hb_blob_destroy (blob);
  return result;
}

static unsigned int
_get_table_tags (const hb_subset_plan_t* plan,
                 unsigned int  start_offset,
                 unsigned int *table_count, /* IN/OUT */
                 hb_tag_t     *table_tags /* OUT */)
{
  unsigned num_tables = hb_face_get_table_tags (plan->source, 0, nullptr, nullptr);
  if (num_tables)
    return hb_face_get_table_tags (plan->source, start_offset, table_count, table_tags);

  // If face has 0 tables associated with it, assume that it was built from
  // hb_face_create_tables and thus is unable to list its tables. Fallback to
  // checking each table type we can handle for existence instead.
  auto it =
      hb_concat (
          + hb_array (known_tables)
          | hb_filter ([&] (hb_tag_t tag) {
            return !_table_is_empty (plan->source, tag) && !plan->no_subset_tables.has (tag);
          })
          | hb_map ([] (hb_tag_t tag) -> hb_tag_t { return tag; }),

          plan->no_subset_tables.iter ()
          | hb_filter([&] (hb_tag_t tag) {
            return !_table_is_empty (plan->source, tag);
          }));

  it += start_offset;

  unsigned num_written = 0;
  while (bool (it) && num_written < *table_count)
    table_tags[num_written++] = *it++;

  *table_count = num_written;
  return num_written;
}


static bool
_is_table_present (hb_face_t *source, hb_tag_t tag)
{

  if (!hb_face_get_table_tags (source, 0, nullptr, nullptr)) {
    // If face has 0 tables associated with it, assume that it was built from
    // hb_face_create_tables and thus is unable to list its tables. Fallback to
    // checking if the blob associated with tag is empty.
    return !_table_is_empty (source, tag);
  }

  hb_tag_t table_tags[32];
  unsigned offset = 0, num_tables = ARRAY_LENGTH (table_tags);
  while (((void) hb_face_get_table_tags (source, offset, &num_tables, table_tags), num_tables))
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
  if (plan->drop_tables.has (tag))
    return true;

  switch (tag)
  {
  case HB_TAG('c','v','a','r'): /* hint table, fallthrough */
    return plan->all_axes_pinned || (plan->flags & HB_SUBSET_FLAGS_NO_HINTING);

  case HB_TAG('c','v','t',' '): /* hint table, fallthrough */
  case HB_TAG('f','p','g','m'): /* hint table, fallthrough */
  case HB_TAG('p','r','e','p'): /* hint table, fallthrough */
  case HB_TAG('h','d','m','x'): /* hint table, fallthrough */
  case HB_TAG('V','D','M','X'): /* hint table, fallthrough */
    return plan->flags & HB_SUBSET_FLAGS_NO_HINTING;

#ifdef HB_NO_SUBSET_LAYOUT
    // Drop Layout Tables if requested.
  case HB_TAG('G','D','E','F'):
  case HB_TAG('G','P','O','S'):
  case HB_TAG('G','S','U','B'):
  case HB_TAG('m','o','r','x'):
  case HB_TAG('m','o','r','t'):
  case HB_TAG('k','e','r','x'):
  case HB_TAG('k','e','r','n'):
    return true;
#endif

  case HB_TAG('a','v','a','r'):
  case HB_TAG('f','v','a','r'):
  case HB_TAG('g','v','a','r'):
  case HB_TAG('H','V','A','R'):
  case HB_TAG('V','V','A','R'):
  case HB_TAG('M','V','A','R'):
    return plan->all_axes_pinned;

  default:
    return false;
  }
}

static bool
_dependencies_satisfied (hb_subset_plan_t *plan, hb_tag_t tag,
                         const hb_set_t &subsetted_tags,
                         const hb_set_t &pending_subset_tags)
{
  switch (tag)
  {
  case HB_TAG('h','m','t','x'):
  case HB_TAG('v','m','t','x'):
  case HB_TAG('m','a','x','p'):
  case HB_TAG('O','S','/','2'):
    return !plan->normalized_coords || !pending_subset_tags.has (HB_TAG('g','l','y','f'));
  case HB_TAG('G','P','O','S'):
    return plan->all_axes_pinned || !pending_subset_tags.has (HB_TAG('G','D','E','F'));
  default:
    return true;
  }
}

static bool
_subset_table (hb_subset_plan_t *plan,
	       hb_vector_t<char> &buf,
	       hb_tag_t tag)
{
  if (plan->no_subset_tables.has (tag)) {
    return _hb_subset_table_passthrough (plan, tag);
  }

  DEBUG_MSG (SUBSET, nullptr, "subset %c%c%c%c", HB_UNTAG (tag));

  bool success;
  if (_hb_subset_table_layout (plan, buf, tag, &success) ||
      _hb_subset_table_var (plan, buf, tag, &success) ||
      _hb_subset_table_cff (plan, buf, tag, &success) ||
      _hb_subset_table_color (plan, buf, tag, &success) ||
      _hb_subset_table_other (plan, buf, tag, &success))
    return success;


  switch (tag)
  {
  case HB_TAG('h','e','a','d'):
    if (_is_table_present (plan->source, HB_TAG('g','l','y','f')) && !_should_drop_table (plan, HB_TAG('g','l','y','f')))
      return true; /* skip head, handled by glyf */
    return _hb_subset_table<const OT::head> (plan, buf);

  case HB_TAG('S','T','A','T'):
    if (!plan->user_axes_location.is_empty ()) return _hb_subset_table<const OT::STAT> (plan, buf);
    else return _hb_subset_table_passthrough (plan, tag);

  case HB_TAG('c','v','t',' '):
#ifndef HB_NO_VAR
    if (_is_table_present (plan->source, HB_TAG('c','v','a','r')) &&
        plan->normalized_coords && !plan->pinned_at_default)
    {
      auto &cvar = *plan->source->table.cvar;
      return OT::cvar::add_cvt_and_apply_deltas (plan, cvar.get_tuple_var_data (), &cvar);
    }
#endif
    return _hb_subset_table_passthrough (plan, tag);
  }

  if (plan->flags & HB_SUBSET_FLAGS_PASSTHROUGH_UNRECOGNIZED)
    return _hb_subset_table_passthrough (plan, tag);

  // Drop table
  return true;
}

static void _attach_accelerator_data (hb_subset_plan_t* plan,
                                      hb_face_t* face /* IN/OUT */)
{
  if (!plan->inprogress_accelerator) return;

  // Transfer the accelerator from the plan to us.
  hb_subset_accelerator_t* accel = plan->inprogress_accelerator;
  plan->inprogress_accelerator = nullptr;

  if (accel->in_error ())
  {
    hb_subset_accelerator_t::destroy (accel);
    return;
  }

  // Populate caches that need access to the final tables.
  hb_blob_ptr_t<OT::cmap> cmap_ptr (hb_sanitize_context_t ().reference_table<OT::cmap> (face));
  accel->cmap_cache = OT::cmap::create_filled_cache (cmap_ptr);
  accel->destroy_cmap_cache = OT::SubtableUnicodesCache::destroy;

  if (!hb_face_set_user_data(face,
                             hb_subset_accelerator_t::user_data_key(),
                             accel,
                             hb_subset_accelerator_t::destroy,
                             true))
    hb_subset_accelerator_t::destroy (accel);
}

/**
 * hb_subset_or_fail:
 * @source: font face data to be subset.
 * @input: input to use for the subsetting.
 *
 * Subsets a font according to provided input. Returns nullptr
 * if the subset operation fails or the face has no glyphs.
 *
 * Since: 2.9.0
 **/
hb_face_t *
hb_subset_or_fail (hb_face_t *source, const hb_subset_input_t *input)
{
  if (unlikely (!input || !source)) return nullptr;

  if (unlikely (!source->get_num_glyphs ()))
  {
    DEBUG_MSG (SUBSET, nullptr, "No glyphs in source font.");
    return nullptr;
  }

  hb_subset_plan_t *plan = hb_subset_plan_create_or_fail (source, input);
  if (unlikely (!plan)) {
    return nullptr;
  }

  hb_face_t * result = hb_subset_plan_execute_or_fail (plan);
  hb_subset_plan_destroy (plan);
  return result;
}


/**
 * hb_subset_plan_execute_or_fail:
 * @plan: a subsetting plan.
 *
 * Executes the provided subsetting @plan.
 *
 * Return value:
 * on success returns a reference to generated font subset. If the subsetting operation fails
 * returns nullptr.
 *
 * Since: 4.0.0
 **/
hb_face_t *
hb_subset_plan_execute_or_fail (hb_subset_plan_t *plan)
{
  if (unlikely (!plan || plan->in_error ())) {
    return nullptr;
  }

  hb_tag_t table_tags[32];
  unsigned offset = 0, num_tables = ARRAY_LENGTH (table_tags);

  hb_set_t subsetted_tags, pending_subset_tags;
  while (((void) _get_table_tags (plan, offset, &num_tables, table_tags), num_tables))
  {
    for (unsigned i = 0; i < num_tables; ++i)
    {
      hb_tag_t tag = table_tags[i];
      if (_should_drop_table (plan, tag)) continue;
      pending_subset_tags.add (tag);
    }

    offset += num_tables;
  }

  bool success = true;

  {
    // Grouping to deallocate buf before calling hb_face_reference (plan->dest).

    hb_vector_t<char> buf;
    buf.alloc (8192 - 16);

    while (!pending_subset_tags.is_empty ())
    {
      if (subsetted_tags.in_error ()
	  || pending_subset_tags.in_error ()) {
	success = false;
	goto end;
      }

      bool made_changes = false;
      for (hb_tag_t tag : pending_subset_tags)
      {
	if (!_dependencies_satisfied (plan, tag,
				      subsetted_tags,
				      pending_subset_tags))
	{
	  // delayed subsetting for some tables since they might have dependency on other tables
	  // in some cases: e.g: during instantiating glyf tables, hmetrics/vmetrics are updated
	  // and saved in subset plan, hmtx/vmtx subsetting need to use these updated metrics values
	  continue;
	}

	pending_subset_tags.del (tag);
	subsetted_tags.add (tag);
	made_changes = true;

	success = _subset_table (plan, buf, tag);
	if (unlikely (!success)) goto end;
      }

      if (!made_changes)
      {
	DEBUG_MSG (SUBSET, nullptr, "Table dependencies unable to be satisfied. Subset failed.");
	success = false;
	goto end;
      }
    }
  }

  if (success && plan->attach_accelerator_data) {
    _attach_accelerator_data (plan, plan->dest);
  }

end:
  return success ? hb_face_reference (plan->dest) : nullptr;
}