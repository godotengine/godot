/*
 * Copyright © 2024  Adobe, Inc.
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
 * Adobe Author(s): Skef Iterum
 */

#include "hb.hh"

#include "hb-depend.hh"

#include "hb-face.hh"
#include "hb-ot-cmap-table.hh"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-math-table.hh"
#include "hb-ot-cff1-table.hh"
#include "OT/Color/COLR/COLR.hh"
#include "OT/Color/COLR/colrv1-depend.hh"

/**
 * SECTION:hb-subset-depend
 * @title: hb-subset-depend
 * @short_description: Glyph dependency graph for font optimization
 * @include: hb-subset-depend.h
 *
 * <warning>This API is highly experimental and subject to change. It is not
 * enabled by default and should not be used in production code without
 * understanding the stability implications.</warning>
 *
 * Functions for extracting a glyph dependency graph.
 *
 * A dependency graph represents the relationships between glyphs in a font,
 * tracking which glyphs reference or produce other glyphs through OpenType
 * mechanisms such as character mapping, glyph substitution, composite
 * construction, color layering, and math variants.
 *
 * Dependency graphs enable finding all glyphs reachable from a given input set.
 * This is useful for font subsetting, analyzing glyph coverage, optimizing
 * font delivery, and determining which glyphs are needed to render specific
 * characters.
 */

hb_subset_depend_t::hb_subset_depend_t (hb_face_t *f)
{
  if (unlikely (!data.glyph_dependencies.resize_exact (f->get_num_glyphs ()))) {
    successful = false;
    return;
  }
  successful = hb_depend_data_builder_t (data).compile (f);
}

bool
hb_depend_data_builder_t::compile (hb_face_t *face)
{
  /* Extract dependencies from all relevant OpenType tables.
   * Note: cmap (UVS) dependencies are not extracted - UVS closure is handled
   * separately via hb_font_get_variation_glyph() query API during closure
   * computation.
   *
   * For GSUB, hb_ot_layout_has_substitution() forces lazy-loader initialization
   * in libharfbuzz, after which get_relaxed() safely returns the already-
   * constructed accelerator without instantiating the GSUB_accelerator_t
   * constructor (which references a hidden is_blocklisted symbol) in this TU. */
  if (hb_ot_layout_has_substitution (face))
    face->table.GSUB.get_relaxed ()->depend (this, face);
#ifndef HB_NO_MATH
  face->table.MATH->depend (this);
#endif
#ifndef HB_NO_COLOR
  face->table.COLR->depend (this);
#endif
  face->table.glyf->depend (this);
#ifndef HB_NO_CFF
  OT::cff1_subset_accelerator_t (face).depend (this);
#endif
  /* XXX TODO: add face->table.VARC->depend (this) here.
   * VARC closure and subsetting are not yet implemented (see hb-subset-plan.cc),
   * so the right traversal architecture for VarComponent records hasn't been
   * established. Implement VARC depend() in the same commit that adds VARC
   * closure, so both share the same traversal model. Component conditions should
   * be treated as over-approximations (include all components regardless of
   * condition), consistent with how FeatureVariations edges are handled. */
  return successful;
}


#ifndef HB_NO_SUBSET_DEPEND

/**
 * hb_subset_depend_from_face_or_fail:
 * @face: font face to collect dependencies from
 *
 * Calculates the dependencies between glyphs in the supplied face.
 * Extracts dependency information from GSUB, glyf, CFF, COLR,
 * and MATH tables. UVS (Unicode Variation Sequence) dependencies
 * are not included; handle those via hb_font_get_variation_glyph().
 *
 * Example:
 * ```c
 * hb_subset_depend_t *depend = hb_subset_depend_from_face_or_fail (face);
 * if (!depend) {
 *   // Handle error (OOM or invalid face)
 *   return;
 * }
 * // ... use depend ...
 * hb_subset_depend_destroy (depend);
 * ```
 *
 * Return value: (transfer full): New depend object, or `nullptr` if creation
 * failed (out of memory or invalid face). Destroy with hb_subset_depend_destroy().
 *
 * Since: 14.3.0
 **/
hb_subset_depend_t *
hb_subset_depend_from_face_or_fail (hb_face_t *face)
{
  hb_subset_depend_t *depend;
  if (unlikely (!(depend = hb_object_create<hb_subset_depend_t> (face))))
    return nullptr;

  if (unlikely (depend->in_error ()))
  {
    hb_subset_depend_destroy (depend);
    return nullptr;
  }

  return depend;
}

/**
 * hb_subset_depend_lookup_glyph:
 * @depend: depend object
 * @gid: GID to retrieve dependencies from
 * @start_offset: offset of first entry to retrieve
 * @entry_count: (inout) (optional): Input = number of entries to fill; output = number
 *   actually filled. Pass NULL to query total count without filling.
 * @entries: (out) (optional) (array length=entry_count): Array to fill with dependency
 *   edge data. May be NULL if @entry_count is also NULL.
 *
 * Retrieve dependency edges for a glyph. Follows the standard HarfBuzz array-getter
 * pattern: always returns the total number of edges for @gid regardless of
 * @start_offset or @entry_count.
 *
 * Example (iterate all entries):
 * ```c
 * unsigned int total = hb_subset_depend_lookup_glyph (depend, gid, 0, NULL, NULL);
 * for (unsigned int i = 0; i < total; i++) {
 *   hb_subset_depend_entry_t entry;
 *   unsigned int count = 1;
 *   hb_subset_depend_lookup_glyph (depend, gid, i, &count, &entry);
 *   // Process entry.table_tag, entry.dependent, etc.
 * }
 * ```
 *
 * Return value: Total number of dependency edges for @gid.
 *
 * Since: 14.3.0
 **/
unsigned int
hb_subset_depend_lookup_glyph (hb_subset_depend_t *depend,
                                hb_codepoint_t gid,
                                unsigned int start_offset,
                                unsigned int *entry_count,
                                hb_subset_depend_entry_t *entries)
{
  unsigned int total = depend->data.get_glyph_entry_count (gid);
  if (entry_count)
  {
    unsigned int count = hb_min (*entry_count,
                                  start_offset < total ? total - start_offset : 0u);
    for (unsigned int i = 0; i < count; i++)
    {
      uint8_t flags_byte = 0;
      depend->data.get_glyph_entry (gid, start_offset + i,
                                    &entries[i].table_tag, &entries[i].dependent,
                                    &entries[i].layout_tag, &entries[i].ligature_set_index,
                                    &entries[i].context_set_index,
                                    &flags_byte);
      entries[i].flags = (hb_subset_depend_edge_flags_t) flags_byte;
    }
    *entry_count = count;
  }
  return total;
}

/**
 * hb_subset_depend_lookup_set:
 * @depend: depend object
 * @index: the index of the set
 * @out: (out): A pointer to a set to copy into
 *
 * Get all glyphs in a set identified by @index.
 * The set index comes from the ligature_set_index or context_set_index field
 * returned by hb_subset_depend_lookup_glyph().
 *
 * Example:
 * ```c
 * hb_set_t *ligature_glyphs = hb_set_create ();
 * if (hb_subset_depend_lookup_set (depend, entry.ligature_set_index, ligature_glyphs)) {
 *   // Process glyphs in the set...
 * }
 * hb_set_destroy (ligature_glyphs);
 * ```
 *
 * Return value: true if there is such a set, false otherwise
 *
 * Since: 14.3.0
 **/
hb_bool_t
hb_subset_depend_lookup_set (hb_subset_depend_t *depend, hb_codepoint_t index,
                              hb_set_t *out)
{
  const hb_set_t *s = depend->get_set_from_index (index);
  if (!s) return false;
  out->set (*s);
  return true;
}

/**
 * hb_subset_depend_destroy:
 * @depend: a #hb_subset_depend_t
 *
 * Decreases the reference count on @depend, and if it reaches zero, destroys
 * @depend, freeing all memory.
 *
 * Since: 14.3.0
 **/
void
hb_subset_depend_destroy (hb_subset_depend_t *depend)
{
  if (!hb_object_destroy (depend)) return;

  hb_free (depend);
}

#endif /* !HB_NO_SUBSET_DEPEND */
