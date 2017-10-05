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

#include "hb-open-type-private.hh"

#include "hb-ot-layout-private.hh"
#include "hb-ot-var-avar-table.hh"
#include "hb-ot-var-fvar-table.hh"
#include "hb-ot-var-mvar-table.hh"
#include "hb-ot-var.h"

/*
 * fvar/avar
 */

static inline const OT::fvar&
_get_fvar (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::fvar);
  hb_ot_layout_t * layout = hb_ot_layout_from_face (face);
  return *(layout->fvar.get ());
}
static inline const OT::avar&
_get_avar (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::avar);
  hb_ot_layout_t * layout = hb_ot_layout_from_face (face);
  return *(layout->avar.get ());
}

/**
 * hb_ot_var_has_data:
 * @face: #hb_face_t to test
 *
 * This function allows to verify the presence of OpenType variation data on the face.
 * Alternatively, use hb_ot_var_get_axis_count().
 *
 * Return value: true if face has a `fvar' table and false otherwise
 *
 * Since: 1.4.2
 **/
hb_bool_t
hb_ot_var_has_data (hb_face_t *face)
{
  return &_get_fvar (face) != &OT::Null(OT::fvar);
}

/**
 * hb_ot_var_get_axis_count:
 *
 * Since: 1.4.2
 **/
unsigned int
hb_ot_var_get_axis_count (hb_face_t *face)
{
  const OT::fvar &fvar = _get_fvar (face);
  return fvar.get_axis_count ();
}

/**
 * hb_ot_var_get_axes:
 *
 * Since: 1.4.2
 **/
unsigned int
hb_ot_var_get_axes (hb_face_t        *face,
		    unsigned int      start_offset,
		    unsigned int     *axes_count /* IN/OUT */,
		    hb_ot_var_axis_t *axes_array /* OUT */)
{
  const OT::fvar &fvar = _get_fvar (face);
  return fvar.get_axis_infos (start_offset, axes_count, axes_array);
}

/**
 * hb_ot_var_find_axis:
 *
 * Since: 1.4.2
 **/
hb_bool_t
hb_ot_var_find_axis (hb_face_t        *face,
		     hb_tag_t          axis_tag,
		     unsigned int     *axis_index,
		     hb_ot_var_axis_t *axis_info)
{
  const OT::fvar &fvar = _get_fvar (face);
  return fvar.find_axis (axis_tag, axis_index, axis_info);
}


/**
 * hb_ot_var_normalize_variations:
 *
 * Since: 1.4.2
 **/
void
hb_ot_var_normalize_variations (hb_face_t            *face,
				const hb_variation_t *variations, /* IN */
				unsigned int          variations_length,
				int                  *coords, /* OUT */
				unsigned int          coords_length)
{
  for (unsigned int i = 0; i < coords_length; i++)
    coords[i] = 0;

  const OT::fvar &fvar = _get_fvar (face);
  for (unsigned int i = 0; i < variations_length; i++)
  {
    unsigned int axis_index;
    if (hb_ot_var_find_axis (face, variations[i].tag, &axis_index, nullptr) &&
	axis_index < coords_length)
      coords[axis_index] = fvar.normalize_axis_value (axis_index, variations[i].value);
  }

  const OT::avar &avar = _get_avar (face);
  avar.map_coords (coords, coords_length);
}

/**
 * hb_ot_var_normalize_coords:
 *
 * Since: 1.4.2
 **/
void
hb_ot_var_normalize_coords (hb_face_t    *face,
			    unsigned int coords_length,
			    const float *design_coords, /* IN */
			    int *normalized_coords /* OUT */)
{
  const OT::fvar &fvar = _get_fvar (face);
  for (unsigned int i = 0; i < coords_length; i++)
    normalized_coords[i] = fvar.normalize_axis_value (i, design_coords[i]);

  const OT::avar &avar = _get_avar (face);
  avar.map_coords (normalized_coords, coords_length);
}
