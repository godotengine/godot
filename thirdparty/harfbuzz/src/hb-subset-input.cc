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

#include "hb-subset-private.hh"
#include "hb-set-private.hh"

/**
 * hb_subset_input_create_or_fail:
 *
 * Return value: New subset input.
 *
 * Since: 1.8.0
 **/
hb_subset_input_t *
hb_subset_input_create_or_fail (void)
{
  hb_subset_input_t *input = hb_object_create<hb_subset_input_t>();

  if (unlikely (!input))
    return nullptr;

  input->unicodes = hb_set_create ();
  input->glyphs = hb_set_create ();
  input->drop_ot_layout = true;

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
hb_subset_input_destroy(hb_subset_input_t *subset_input)
{
  if (!hb_object_destroy (subset_input)) return;

  hb_set_destroy (subset_input->unicodes);
  hb_set_destroy (subset_input->glyphs);

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

/**
 * hb_subset_input_drop_hints:
 * @subset_input: a subset_input.
 *
 * Since: 1.8.0
 **/
HB_EXTERN hb_bool_t *
hb_subset_input_drop_hints (hb_subset_input_t *subset_input)
{
  return &subset_input->drop_hints;
}

/**
 * hb_subset_input_drop_ot_layout:
 * @subset_input: a subset_input.
 *
 * If enabled ot layout tables will be dropped as part of
 * the subsetting operation. Currently this defaults to
 * true.
 *
 * Since: REPLACEME
 **/
HB_EXTERN hb_bool_t *
hb_subset_input_drop_ot_layout (hb_subset_input_t *subset_input)
{
  return &subset_input->drop_ot_layout;
}
