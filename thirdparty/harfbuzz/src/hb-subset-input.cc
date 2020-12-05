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

#include "hb-subset.hh"
#include "hb-set.hh"

/**
 * hb_subset_input_create_or_fail:
 *
 * Return value: New subset input.
 *
 * Since: 1.8.0
 **/
hb_subset_input_t *
hb_subset_input_create_or_fail ()
{
  hb_subset_input_t *input = hb_object_create<hb_subset_input_t>();

  if (unlikely (!input))
    return nullptr;

  input->unicodes = hb_set_create ();
  input->glyphs = hb_set_create ();
  input->name_ids = hb_set_create ();
  hb_set_add_range (input->name_ids, 0, 6);
  input->name_languages = hb_set_create ();
  hb_set_add (input->name_languages, 0x0409);
  input->drop_tables = hb_set_create ();
  input->drop_hints = false;
  input->desubroutinize = false;
  input->retain_gids = false;
  input->name_legacy = false;

  hb_tag_t default_drop_tables[] = {
    // Layout disabled by default
    HB_TAG ('G', 'S', 'U', 'B'),
    HB_TAG ('G', 'P', 'O', 'S'),
    HB_TAG ('G', 'D', 'E', 'F'),
    HB_TAG ('m', 'o', 'r', 'x'),
    HB_TAG ('m', 'o', 'r', 't'),
    HB_TAG ('k', 'e', 'r', 'x'),
    HB_TAG ('k', 'e', 'r', 'n'),

    // Copied from fontTools:
    HB_TAG ('B', 'A', 'S', 'E'),
    HB_TAG ('J', 'S', 'T', 'F'),
    HB_TAG ('D', 'S', 'I', 'G'),
    HB_TAG ('E', 'B', 'D', 'T'),
    HB_TAG ('E', 'B', 'L', 'C'),
    HB_TAG ('E', 'B', 'S', 'C'),
    HB_TAG ('S', 'V', 'G', ' '),
    HB_TAG ('P', 'C', 'L', 'T'),
    HB_TAG ('L', 'T', 'S', 'H'),
    // Graphite tables
    HB_TAG ('F', 'e', 'a', 't'),
    HB_TAG ('G', 'l', 'a', 't'),
    HB_TAG ('G', 'l', 'o', 'c'),
    HB_TAG ('S', 'i', 'l', 'f'),
    HB_TAG ('S', 'i', 'l', 'l'),
  };

  input->drop_tables->add_array (default_drop_tables, ARRAY_LENGTH (default_drop_tables));

  return input;
}

/**
 * hb_subset_input_reference: (skip)
 * @subset_input: a subset_input.
 *
 *
 *
 * Return value:
 *
 * Since: 1.8.0
 **/
hb_subset_input_t *
hb_subset_input_reference (hb_subset_input_t *subset_input)
{
  return hb_object_reference (subset_input);
}

/**
 * hb_subset_input_destroy:
 * @subset_input: a subset_input.
 *
 * Since: 1.8.0
 **/
void
hb_subset_input_destroy (hb_subset_input_t *subset_input)
{
  if (!hb_object_destroy (subset_input)) return;

  hb_set_destroy (subset_input->unicodes);
  hb_set_destroy (subset_input->glyphs);
  hb_set_destroy (subset_input->name_ids);
  hb_set_destroy (subset_input->name_languages);
  hb_set_destroy (subset_input->drop_tables);

  free (subset_input);
}

/**
 * hb_subset_input_unicode_set:
 * @subset_input: a subset_input.
 *
 * Since: 1.8.0
 **/
HB_EXTERN hb_set_t *
hb_subset_input_unicode_set (hb_subset_input_t *subset_input)
{
  return subset_input->unicodes;
}

/**
 * hb_subset_input_glyph_set:
 * @subset_input: a subset_input.
 *
 * Since: 1.8.0
 **/
HB_EXTERN hb_set_t *
hb_subset_input_glyph_set (hb_subset_input_t *subset_input)
{
  return subset_input->glyphs;
}

HB_EXTERN hb_set_t *
hb_subset_input_nameid_set (hb_subset_input_t *subset_input)
{
  return subset_input->name_ids;
}

HB_EXTERN hb_set_t *
hb_subset_input_namelangid_set (hb_subset_input_t *subset_input)
{
  return subset_input->name_languages;
}

HB_EXTERN hb_set_t *
hb_subset_input_drop_tables_set (hb_subset_input_t *subset_input)
{
  return subset_input->drop_tables;
}

HB_EXTERN void
hb_subset_input_set_drop_hints (hb_subset_input_t *subset_input,
				hb_bool_t drop_hints)
{
  subset_input->drop_hints = drop_hints;
}

HB_EXTERN hb_bool_t
hb_subset_input_get_drop_hints (hb_subset_input_t *subset_input)
{
  return subset_input->drop_hints;
}

HB_EXTERN void
hb_subset_input_set_desubroutinize (hb_subset_input_t *subset_input,
				    hb_bool_t desubroutinize)
{
  subset_input->desubroutinize = desubroutinize;
}

HB_EXTERN hb_bool_t
hb_subset_input_get_desubroutinize (hb_subset_input_t *subset_input)
{
  return subset_input->desubroutinize;
}

/**
 * hb_subset_input_set_retain_gids:
 * @subset_input: a subset_input.
 * @retain_gids: If true the subsetter will not renumber glyph ids.
 * Since: 2.4.0
 **/
HB_EXTERN void
hb_subset_input_set_retain_gids (hb_subset_input_t *subset_input,
				 hb_bool_t retain_gids)
{
  subset_input->retain_gids = retain_gids;
}

/**
 * hb_subset_input_get_retain_gids:
 * Returns: value of retain_gids.
 * Since: 2.4.0
 **/
HB_EXTERN hb_bool_t
hb_subset_input_get_retain_gids (hb_subset_input_t *subset_input)
{
  return subset_input->retain_gids;
}

HB_EXTERN void
hb_subset_input_set_name_legacy (hb_subset_input_t *subset_input,
				 hb_bool_t name_legacy)
{
  subset_input->name_legacy = name_legacy;
}

HB_EXTERN hb_bool_t
hb_subset_input_get_name_legacy (hb_subset_input_t *subset_input)
{
  return subset_input->name_legacy;
}
