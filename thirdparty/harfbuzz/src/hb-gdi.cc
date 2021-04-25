/*
 * Copyright © 2019  Ebrahim Byagowi
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
 */

#include "hb.hh"

#ifdef HAVE_GDI

#include "hb-gdi.h"


/**
 * SECTION:hb-gdi
 * @title: hb-gdi
 * @short_description: GDI integration
 * @include: hb-gdi.h
 *
 * Functions for using HarfBuzz with GDI fonts.
 **/

static hb_blob_t *
_hb_gdi_reference_table (hb_face_t *face HB_UNUSED, hb_tag_t tag, void *user_data)
{
  char *buffer = nullptr;
  DWORD length = 0;

  HDC hdc = GetDC (nullptr);
  if (unlikely (!SelectObject (hdc, (HFONT) user_data))) goto fail;

  length = GetFontData (hdc, hb_uint32_swap (tag), 0, buffer, length);
  if (unlikely (length == GDI_ERROR)) goto fail_with_releasedc;

  buffer = (char *) malloc (length);
  if (unlikely (!buffer)) goto fail_with_releasedc;
  length = GetFontData (hdc, hb_uint32_swap (tag), 0, buffer, length);
  if (unlikely (length == GDI_ERROR)) goto fail_with_releasedc_and_free;
  ReleaseDC (nullptr, hdc);

  return hb_blob_create ((const char *) buffer, length, HB_MEMORY_MODE_WRITABLE, buffer, free);

fail_with_releasedc_and_free:
  free (buffer);
fail_with_releasedc:
  ReleaseDC (nullptr, hdc);
fail:
  return hb_blob_get_empty ();
}

/**
 * hb_gdi_face_create:
 * @hfont: a HFONT object.
 *
 * Constructs a new face object from the specified GDI HFONT.
 *
 * Return value: #hb_face_t object corresponding to the given input
 *
 * Since: 2.6.0
 **/
hb_face_t *
hb_gdi_face_create (HFONT hfont)
{
  return hb_face_create_for_tables (_hb_gdi_reference_table, (void *) hfont, nullptr);
}

#endif
