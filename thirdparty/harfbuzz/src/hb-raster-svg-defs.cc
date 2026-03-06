/*
 * Copyright © 2026  Behdad Esfahbod
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
 * Author(s): Behdad Esfahbod
 */

#ifndef HB_NO_RASTER_SVG

#include "hb.hh"

#include "hb-raster-svg-defs.hh"

#include <string.h>

hb_svg_defs_t::~hb_svg_defs_t ()
{
  for (unsigned i = 0; i < owned_id_strings.length; i++)
    hb_free (owned_id_strings.arrayZ[i]);
}

bool
hb_svg_defs_t::add_id_mapping (hb_hashmap_t<hb_bytes_t, unsigned> *map,
                               hb_bytes_t id,
                               unsigned idx)
{
  if (!id.length)
    return false;
  if (map->has (id))
    return true;

  unsigned n = (unsigned) id.length;
  char *owned = (char *) hb_malloc (n + 1);
  if (unlikely (!owned))
    return false;
  hb_memcpy (owned, id.arrayZ, n);
  owned[n] = '\0';
  if (unlikely (!owned_id_strings.push (owned)))
  {
    hb_free (owned);
    return false;
  }

  hb_bytes_t owned_key = hb_bytes_t (owned, n);
  if (unlikely (!map->set (owned_key, idx)))
    return false;
  return true;
}

bool
hb_svg_defs_t::add_gradient (hb_bytes_t id, const hb_svg_gradient_t &grad)
{
  unsigned idx = gradients.length;
  gradients.push (grad);
  if (unlikely (gradients.in_error ()))
    return false;

  if (unlikely (!add_id_mapping (&gradient_by_id, id, idx)))
  {
    gradients.pop ();
    return false;
  }
  return true;
}

const hb_svg_gradient_t *
hb_svg_defs_t::find_gradient (hb_bytes_t id) const
{
  if (!id.length) return nullptr;
  unsigned *idx = nullptr;
  if (gradient_by_id.has (id, &idx))
    return &gradients[*idx];
  return nullptr;
}

bool
hb_svg_defs_t::add_clip_path (hb_bytes_t id, const hb_svg_clip_path_def_t &clip)
{
  unsigned idx = clip_paths.length;
  clip_paths.push (clip);
  if (unlikely (clip_paths.in_error ()))
    return false;

  if (unlikely (!add_id_mapping (&clip_path_by_id, id, idx)))
  {
    clip_paths.pop ();
    return false;
  }
  return true;
}

const hb_svg_clip_path_def_t *
hb_svg_defs_t::find_clip_path (hb_bytes_t id) const
{
  if (!id.length) return nullptr;
  unsigned *idx = nullptr;
  if (clip_path_by_id.has (id, &idx))
    return &clip_paths[*idx];
  return nullptr;
}

#endif /* !HB_NO_RASTER_SVG */
