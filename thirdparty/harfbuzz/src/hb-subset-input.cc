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
 * Creates a new subset input object.
 *
 * Return value: (transfer full): New subset input, or `NULL` if failed. Destroy
 * with hb_subset_input_destroy().
 *
 * Since: 1.8.0
 **/
hb_subset_input_t *
hb_subset_input_create_or_fail (void)
{
  hb_subset_input_t *input = hb_object_create<hb_subset_input_t>();

  if (unlikely (!input))
    return nullptr;

  for (auto& set : input->sets_iter ())
    set = hb_set_create ();

  input->axes_location = hb_hashmap_create<hb_tag_t, float> ();
  
  if (!input->axes_location || input->in_error ())
  {
    hb_subset_input_destroy (input);
    return nullptr;
  }

  input->flags = HB_SUBSET_FLAGS_DEFAULT;

  hb_set_add_range (input->sets.name_ids, 0, 6);
  hb_set_add (input->sets.name_languages, 0x0409);

  hb_tag_t default_drop_tables[] = {
    // Layout disabled by default
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
  input->sets.drop_tables->add_array (default_drop_tables, ARRAY_LENGTH (default_drop_tables));

  hb_tag_t default_no_subset_tables[] = {
    HB_TAG ('a', 'v', 'a', 'r'),
    HB_TAG ('f', 'v', 'a', 'r'),
    HB_TAG ('g', 'a', 's', 'p'),
    HB_TAG ('c', 'v', 't', ' '),
    HB_TAG ('f', 'p', 'g', 'm'),
    HB_TAG ('p', 'r', 'e', 'p'),
    HB_TAG ('V', 'D', 'M', 'X'),
    HB_TAG ('D', 'S', 'I', 'G'),
    HB_TAG ('M', 'V', 'A', 'R'),
    HB_TAG ('c', 'v', 'a', 'r'),
  };
  input->sets.no_subset_tables->add_array (default_no_subset_tables,
                                         ARRAY_LENGTH (default_no_subset_tables));

  //copied from _layout_features_groups in fonttools
  hb_tag_t default_layout_features[] = {
    // default shaper
    // common
    HB_TAG ('r', 'v', 'r', 'n'),
    HB_TAG ('c', 'c', 'm', 'p'),
    HB_TAG ('l', 'i', 'g', 'a'),
    HB_TAG ('l', 'o', 'c', 'l'),
    HB_TAG ('m', 'a', 'r', 'k'),
    HB_TAG ('m', 'k', 'm', 'k'),
    HB_TAG ('r', 'l', 'i', 'g'),

    //fractions
    HB_TAG ('f', 'r', 'a', 'c'),
    HB_TAG ('n', 'u', 'm', 'r'),
    HB_TAG ('d', 'n', 'o', 'm'),

    //horizontal
    HB_TAG ('c', 'a', 'l', 't'),
    HB_TAG ('c', 'l', 'i', 'g'),
    HB_TAG ('c', 'u', 'r', 's'),
    HB_TAG ('k', 'e', 'r', 'n'),
    HB_TAG ('r', 'c', 'l', 't'),

    //vertical
    HB_TAG ('v', 'a', 'l', 't'),
    HB_TAG ('v', 'e', 'r', 't'),
    HB_TAG ('v', 'k', 'r', 'n'),
    HB_TAG ('v', 'p', 'a', 'l'),
    HB_TAG ('v', 'r', 't', '2'),

    //ltr
    HB_TAG ('l', 't', 'r', 'a'),
    HB_TAG ('l', 't', 'r', 'm'),

    //rtl
    HB_TAG ('r', 't', 'l', 'a'),
    HB_TAG ('r', 't', 'l', 'm'),

    //random
    HB_TAG ('r', 'a', 'n', 'd'),

    //justify
    HB_TAG ('j', 'a', 'l', 't'), // HarfBuzz doesn't use; others might

    //private
    HB_TAG ('H', 'a', 'r', 'f'),
    HB_TAG ('H', 'A', 'R', 'F'),
    HB_TAG ('B', 'u', 'z', 'z'),
    HB_TAG ('B', 'U', 'Z', 'Z'),

    //shapers

    //arabic
    HB_TAG ('i', 'n', 'i', 't'),
    HB_TAG ('m', 'e', 'd', 'i'),
    HB_TAG ('f', 'i', 'n', 'a'),
    HB_TAG ('i', 's', 'o', 'l'),
    HB_TAG ('m', 'e', 'd', '2'),
    HB_TAG ('f', 'i', 'n', '2'),
    HB_TAG ('f', 'i', 'n', '3'),
    HB_TAG ('c', 's', 'w', 'h'),
    HB_TAG ('m', 's', 'e', 't'),
    HB_TAG ('s', 't', 'c', 'h'),

    //hangul
    HB_TAG ('l', 'j', 'm', 'o'),
    HB_TAG ('v', 'j', 'm', 'o'),
    HB_TAG ('t', 'j', 'm', 'o'),

    //tibetan
    HB_TAG ('a', 'b', 'v', 's'),
    HB_TAG ('b', 'l', 'w', 's'),
    HB_TAG ('a', 'b', 'v', 'm'),
    HB_TAG ('b', 'l', 'w', 'm'),

    //indic
    HB_TAG ('n', 'u', 'k', 't'),
    HB_TAG ('a', 'k', 'h', 'n'),
    HB_TAG ('r', 'p', 'h', 'f'),
    HB_TAG ('r', 'k', 'r', 'f'),
    HB_TAG ('p', 'r', 'e', 'f'),
    HB_TAG ('b', 'l', 'w', 'f'),
    HB_TAG ('h', 'a', 'l', 'f'),
    HB_TAG ('a', 'b', 'v', 'f'),
    HB_TAG ('p', 's', 't', 'f'),
    HB_TAG ('c', 'f', 'a', 'r'),
    HB_TAG ('v', 'a', 't', 'u'),
    HB_TAG ('c', 'j', 'c', 't'),
    HB_TAG ('i', 'n', 'i', 't'),
    HB_TAG ('p', 'r', 'e', 's'),
    HB_TAG ('a', 'b', 'v', 's'),
    HB_TAG ('b', 'l', 'w', 's'),
    HB_TAG ('p', 's', 't', 's'),
    HB_TAG ('h', 'a', 'l', 'n'),
    HB_TAG ('d', 'i', 's', 't'),
    HB_TAG ('a', 'b', 'v', 'm'),
    HB_TAG ('b', 'l', 'w', 'm'),
  };

  input->sets.layout_features->add_array (default_layout_features, ARRAY_LENGTH (default_layout_features));

  input->sets.layout_scripts->invert (); // Default to all scripts.

  if (input->in_error ())
  {
    hb_subset_input_destroy (input);
    return nullptr;
  }
  return input;
}

/**
 * hb_subset_input_reference: (skip)
 * @input: a #hb_subset_input_t object.
 *
 * Increases the reference count on @input.
 *
 * Return value: @input.
 *
 * Since: 1.8.0
 **/
hb_subset_input_t *
hb_subset_input_reference (hb_subset_input_t *input)
{
  return hb_object_reference (input);
}

/**
 * hb_subset_input_destroy:
 * @input: a #hb_subset_input_t object.
 *
 * Decreases the reference count on @input, and if it reaches zero, destroys
 * @input, freeing all memory.
 *
 * Since: 1.8.0
 **/
void
hb_subset_input_destroy (hb_subset_input_t *input)
{
  if (!hb_object_destroy (input)) return;

  for (hb_set_t* set : input->sets_iter ())
    hb_set_destroy (set);

  hb_hashmap_destroy (input->axes_location);

  hb_free (input);
}

/**
 * hb_subset_input_unicode_set:
 * @input: a #hb_subset_input_t object.
 *
 * Gets the set of Unicode code points to retain, the caller should modify the
 * set as needed.
 *
 * Return value: (transfer none): pointer to the #hb_set_t of Unicode code
 * points.
 *
 * Since: 1.8.0
 **/
HB_EXTERN hb_set_t *
hb_subset_input_unicode_set (hb_subset_input_t *input)
{
  return input->sets.unicodes;
}

/**
 * hb_subset_input_glyph_set:
 * @input: a #hb_subset_input_t object.
 *
 * Gets the set of glyph IDs to retain, the caller should modify the set as
 * needed.
 *
 * Return value: (transfer none): pointer to the #hb_set_t of glyph IDs.
 *
 * Since: 1.8.0
 **/
HB_EXTERN hb_set_t *
hb_subset_input_glyph_set (hb_subset_input_t *input)
{
  return input->sets.glyphs;
}

/**
 * hb_subset_input_set:
 * @input: a #hb_subset_input_t object.
 * @set_type: a #hb_subset_sets_t set type.
 *
 * Gets the set of the specified type.
 *
 * Return value: (transfer none): pointer to the #hb_set_t of the specified type.
 *
 * Since: 2.9.1
 **/
HB_EXTERN hb_set_t *
hb_subset_input_set (hb_subset_input_t *input, hb_subset_sets_t set_type)
{
  return input->sets_iter () [set_type];
}

/**
 * hb_subset_input_get_flags:
 * @input: a #hb_subset_input_t object.
 *
 * Gets all of the subsetting flags in the input object.
 *
 * Return value: the subsetting flags bit field.
 *
 * Since: 2.9.0
 **/
HB_EXTERN hb_subset_flags_t
hb_subset_input_get_flags (hb_subset_input_t *input)
{
  return (hb_subset_flags_t) input->flags;
}

/**
 * hb_subset_input_set_flags:
 * @input: a #hb_subset_input_t object.
 * @value: bit field of flags
 *
 * Sets all of the flags in the input object to the values specified by the bit
 * field.
 *
 * Since: 2.9.0
 **/
HB_EXTERN void
hb_subset_input_set_flags (hb_subset_input_t *input,
			   unsigned value)
{
  input->flags = (hb_subset_flags_t) value;
}

/**
 * hb_subset_input_set_user_data: (skip)
 * @input: a #hb_subset_input_t object.
 * @key: The user-data key to set
 * @data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 * @replace: Whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the given subset input object.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 2.9.0
 **/
hb_bool_t
hb_subset_input_set_user_data (hb_subset_input_t  *input,
			       hb_user_data_key_t *key,
			       void *		   data,
			       hb_destroy_func_t   destroy,
			       hb_bool_t	   replace)
{
  return hb_object_set_user_data (input, key, data, destroy, replace);
}

/**
 * hb_subset_input_get_user_data: (skip)
 * @input: a #hb_subset_input_t object.
 * @key: The user-data key to query
 *
 * Fetches the user data associated with the specified key,
 * attached to the specified subset input object.
 *
 * Return value: (transfer none): A pointer to the user data
 *
 * Since: 2.9.0
 **/
void *
hb_subset_input_get_user_data (const hb_subset_input_t *input,
			       hb_user_data_key_t     *key)
{
  return hb_object_get_user_data (input, key);
}

#ifdef HB_EXPERIMENTAL_API
#ifndef HB_NO_VAR
/**
 * hb_subset_input_pin_axis_to_default: (skip)
 * @input: a #hb_subset_input_t object.
 * @axis_tag: Tag of the axis to be pinned
 *
 * Pin an axis to its default location in the given subset input object.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: REPLACEME
 **/
hb_bool_t
hb_subset_input_pin_axis_to_default (hb_subset_input_t  *input,
                                     hb_face_t          *face,
                                     hb_tag_t            axis_tag)
{
  hb_ot_var_axis_info_t axis_info;
  if (!hb_ot_var_find_axis_info (face, axis_tag, &axis_info))
    return false;

  return input->axes_location->set (axis_tag, axis_info.default_value);
}

/**
 * hb_subset_input_pin_axis_location: (skip)
 * @input: a #hb_subset_input_t object.
 * @axis_tag: Tag of the axis to be pinned
 * @axis_value: Location on the axis to be pinned at
 *
 * Pin an axis to a fixed location in the given subset input object.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: REPLACEME
 **/
hb_bool_t
hb_subset_input_pin_axis_location (hb_subset_input_t  *input,
                                   hb_face_t          *face,
                                   hb_tag_t            axis_tag,
                                   float               axis_value)
{
  hb_ot_var_axis_info_t axis_info;
  if (!hb_ot_var_find_axis_info (face, axis_tag, &axis_info))
    return false;

  float val = hb_clamp(axis_value, axis_info.min_value, axis_info.max_value);
  return input->axes_location->set (axis_tag, val);
}
#endif
#endif
