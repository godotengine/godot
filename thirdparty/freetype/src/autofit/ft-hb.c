/*
 * Copyright © 2009, 2023  Red Hat, Inc.
 * Copyright © 2015  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod, Matthias Clasen
 * Google Author(s): Behdad Esfahbod
 */

#include <freetype/freetype.h>
#include <freetype/tttables.h>

#ifdef FT_CONFIG_OPTION_USE_HARFBUZZ

#include "ft-hb.h"

/* The following three functions are a more or less verbatim
 * copy of corresponding HarfBuzz code from hb-ft.cc
 */

static hb_blob_t *
hb_ft_reference_table_ (hb_face_t *face, hb_tag_t tag, void *user_data)
{
  FT_Face ft_face = (FT_Face) user_data;
  FT_Byte *buffer;
  FT_ULong  length = 0;
  FT_Error error;

  FT_UNUSED (face);

  /* Note: FreeType like HarfBuzz uses the NONE tag for fetching the entire blob */

  error = FT_Load_Sfnt_Table (ft_face, tag, 0, NULL, &length);
  if (error)
    return NULL;

  buffer = (FT_Byte *) ft_smalloc (length);
  if (!buffer)
    return NULL;

  error = FT_Load_Sfnt_Table (ft_face, tag, 0, buffer, &length);
  if (error)
  {
    free (buffer);
    return NULL;
  }

  return hb_blob_create ((const char *) buffer, length,
                         HB_MEMORY_MODE_WRITABLE,
                         buffer, ft_sfree);
}

static hb_face_t *
hb_ft_face_create_ (FT_Face           ft_face,
                    hb_destroy_func_t destroy)
{
  hb_face_t *face;

  if (!ft_face->stream->read) {
    hb_blob_t *blob;

    blob = hb_blob_create ((const char *) ft_face->stream->base,
                           (unsigned int) ft_face->stream->size,
                           HB_MEMORY_MODE_READONLY,
                           ft_face, destroy);
    face = hb_face_create (blob, ft_face->face_index);
    hb_blob_destroy (blob);
  } else {
    face = hb_face_create_for_tables (hb_ft_reference_table_, ft_face, destroy);
  }

  hb_face_set_index (face, ft_face->face_index);
  hb_face_set_upem (face, ft_face->units_per_EM);

  return face;
}

FT_LOCAL_DEF(hb_font_t *)
hb_ft_font_create_ (FT_Face           ft_face,
                    hb_destroy_func_t destroy)
{
  hb_font_t *font;
  hb_face_t *face;

  face = hb_ft_face_create_ (ft_face, destroy);
  font = hb_font_create (face);
  hb_face_destroy (face);
  return font;
}

#else /* !FT_CONFIG_OPTION_USE_HARFBUZZ */

/* ANSI C doesn't like empty source files */
typedef int  _ft_hb_dummy;

#endif /* !FT_CONFIG_OPTION_USE_HARFBUZZ */

/* END */
