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
 * Google Author(s): Garret Rieger, Roderick Sheeter
 */

#include "hb-map-private.hh"
#include "hb-subset-private.hh"
#include "hb-set-private.hh"

#include "hb-subset-plan.hh"
#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"

static void
_add_gid_and_children (const OT::glyf::accelerator_t &glyf,
		       hb_codepoint_t gid,
		       hb_set_t *gids_to_retain)
{
  if (hb_set_has (gids_to_retain, gid))
    // Already visited this gid, ignore.
    return;

  hb_set_add (gids_to_retain, gid);

  OT::glyf::CompositeGlyphHeader::Iterator composite;
  if (glyf.get_composite (gid, &composite))
  {
    do
    {
      _add_gid_and_children (glyf, (hb_codepoint_t) composite.current->glyphIndex, gids_to_retain);
    } while (composite.move_to_next());
  }
}

static void
_gsub_closure (hb_face_t *face, hb_set_t *gids_to_retain)
{
  hb_auto_t<hb_set_t> lookup_indices;
  hb_ot_layout_collect_lookups (face,
                                HB_OT_TAG_GSUB,
                                nullptr,
                                nullptr,
                                nullptr,
                                &lookup_indices);
  hb_ot_layout_lookups_substitute_closure (face,
                                           &lookup_indices,
                                           gids_to_retain);
}


static void
_populate_gids_to_retain (hb_face_t *face,
                          const hb_set_t *unicodes,
                          bool close_over_gsub,
                          hb_set_t *unicodes_to_retain,
                          hb_map_t *codepoint_to_glyph,
                          hb_vector_t<hb_codepoint_t> *glyphs)
{
  OT::cmap::accelerator_t cmap;
  OT::glyf::accelerator_t glyf;
  cmap.init (face);
  glyf.init (face);

  hb_set_t *initial_gids_to_retain = hb_set_create ();
  initial_gids_to_retain->add (0); // Not-def

  hb_codepoint_t cp = HB_SET_VALUE_INVALID;
  while (unicodes->next (&cp))
  {
    hb_codepoint_t gid;
    if (!cmap.get_nominal_glyph (cp, &gid))
    {
      DEBUG_MSG(SUBSET, nullptr, "Drop U+%04X; no gid", cp);
      continue;
    }
    unicodes_to_retain->add (cp);
    codepoint_to_glyph->set (cp, gid);
    initial_gids_to_retain->add (gid);
  }

  if (close_over_gsub)
    // Add all glyphs needed for GSUB substitutions.
    _gsub_closure (face, initial_gids_to_retain);

  // Populate a full set of glyphs to retain by adding all referenced
  // composite glyphs.
  hb_codepoint_t gid = HB_SET_VALUE_INVALID;
  hb_set_t *all_gids_to_retain = hb_set_create ();
  while (initial_gids_to_retain->next (&gid))
  {
    _add_gid_and_children (glyf, gid, all_gids_to_retain);
  }
  hb_set_destroy (initial_gids_to_retain);

  glyphs->alloc (all_gids_to_retain->get_population ());
  gid = HB_SET_VALUE_INVALID;
  while (all_gids_to_retain->next (&gid))
    glyphs->push (gid);

  hb_set_destroy (all_gids_to_retain);
  glyf.fini ();
  cmap.fini ();
}

static void
_create_old_gid_to_new_gid_map (const hb_vector_t<hb_codepoint_t> &glyphs,
                                hb_map_t *glyph_map)
{
  for (unsigned int i = 0; i < glyphs.len; i++) {
    glyph_map->set (glyphs[i], i);
  }
}

/**
 * hb_subset_plan_create:
 * Computes a plan for subsetting the supplied face according
 * to a provide profile and input. The plan describes
 * which tables and glyphs should be retained.
 *
 * Return value: New subset plan.
 *
 * Since: 1.7.5
 **/
hb_subset_plan_t *
hb_subset_plan_create (hb_face_t           *face,
                       hb_subset_profile_t *profile,
                       hb_subset_input_t   *input)
{
  hb_subset_plan_t *plan = hb_object_create<hb_subset_plan_t> ();

  plan->drop_hints = input->drop_hints;
  plan->drop_ot_layout = input->drop_ot_layout;
  plan->unicodes = hb_set_create();
  plan->glyphs.init();
  plan->source = hb_face_reference (face);
  plan->dest = hb_subset_face_create ();
  plan->codepoint_to_glyph = hb_map_create();
  plan->glyph_map = hb_map_create();

  _populate_gids_to_retain (face,
                            input->unicodes,
                            !plan->drop_ot_layout,
                            plan->unicodes,
                            plan->codepoint_to_glyph,
                            &plan->glyphs);
  _create_old_gid_to_new_gid_map (plan->glyphs,
                                  plan->glyph_map);

  return plan;
}

/**
 * hb_subset_plan_destroy:
 *
 * Since: 1.7.5
 **/
void
hb_subset_plan_destroy (hb_subset_plan_t *plan)
{
  if (!hb_object_destroy (plan)) return;

  hb_set_destroy (plan->unicodes);
  plan->glyphs.fini();
  hb_face_destroy (plan->source);
  hb_face_destroy (plan->dest);
  hb_map_destroy (plan->codepoint_to_glyph);
  hb_map_destroy (plan->glyph_map);

  free (plan);
}
