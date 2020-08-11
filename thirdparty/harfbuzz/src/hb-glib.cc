/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2011  Google, Inc.
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
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifdef HAVE_GLIB

#include "hb-glib.h"

#include "hb-machinery.hh"


/**
 * SECTION:hb-glib
 * @title: hb-glib
 * @short_description: GLib integration
 * @include: hb-glib.h
 *
 * Functions for using HarfBuzz with the GLib library.
 *
 * HarfBuzz supports using GLib to provide Unicode data, by attaching
 * GLib functions to the virtual methods in a #hb_unicode_funcs_t function
 * structure.
 **/


/**
 * hb_glib_script_to_script:
 * @script: The GUnicodeScript identifier to query
 *
 * Fetches the #hb_script_t script that corresponds to the
 * specified GUnicodeScript identifier.
 *
 * Return value: the #hb_script_t script found
 *
 * Since: 0.9.38
 **/
hb_script_t
hb_glib_script_to_script (GUnicodeScript script)
{
  return (hb_script_t) g_unicode_script_to_iso15924 (script);
}

/**
 * hb_glib_script_from_script:
 * @script: The #hb_script_t to query
 *
 * Fetches the GUnicodeScript identifier that corresponds to the
 * specified #hb_script_t script.
 *
 * Return value: the GUnicodeScript identifier found
 *
 * Since: 0.9.38
 **/
GUnicodeScript
hb_glib_script_from_script (hb_script_t script)
{
  return g_unicode_script_from_iso15924 (script);
}


static hb_unicode_combining_class_t
hb_glib_unicode_combining_class (hb_unicode_funcs_t *ufuncs HB_UNUSED,
				 hb_codepoint_t      unicode,
				 void               *user_data HB_UNUSED)

{
  return (hb_unicode_combining_class_t) g_unichar_combining_class (unicode);
}

static hb_unicode_general_category_t
hb_glib_unicode_general_category (hb_unicode_funcs_t *ufuncs HB_UNUSED,
				  hb_codepoint_t      unicode,
				  void               *user_data HB_UNUSED)

{
  /* hb_unicode_general_category_t and GUnicodeType are identical */
  return (hb_unicode_general_category_t) g_unichar_type (unicode);
}

static hb_codepoint_t
hb_glib_unicode_mirroring (hb_unicode_funcs_t *ufuncs HB_UNUSED,
			   hb_codepoint_t      unicode,
			   void               *user_data HB_UNUSED)
{
  g_unichar_get_mirror_char (unicode, &unicode);
  return unicode;
}

static hb_script_t
hb_glib_unicode_script (hb_unicode_funcs_t *ufuncs HB_UNUSED,
			hb_codepoint_t      unicode,
			void               *user_data HB_UNUSED)
{
  return hb_glib_script_to_script (g_unichar_get_script (unicode));
}

static hb_bool_t
hb_glib_unicode_compose (hb_unicode_funcs_t *ufuncs HB_UNUSED,
			 hb_codepoint_t      a,
			 hb_codepoint_t      b,
			 hb_codepoint_t     *ab,
			 void               *user_data HB_UNUSED)
{
#if GLIB_CHECK_VERSION(2,29,12)
  return g_unichar_compose (a, b, ab);
#endif

  /* We don't ifdef-out the fallback code such that compiler always
   * sees it and makes sure it's compilable. */

  gchar utf8[12];
  gchar *normalized;
  int len;
  hb_bool_t ret;

  len = g_unichar_to_utf8 (a, utf8);
  len += g_unichar_to_utf8 (b, utf8 + len);
  normalized = g_utf8_normalize (utf8, len, G_NORMALIZE_NFC);
  len = g_utf8_strlen (normalized, -1);
  if (unlikely (!len))
    return false;

  if (len == 1) {
    *ab = g_utf8_get_char (normalized);
    ret = true;
  } else {
    ret = false;
  }

  g_free (normalized);
  return ret;
}

static hb_bool_t
hb_glib_unicode_decompose (hb_unicode_funcs_t *ufuncs HB_UNUSED,
			   hb_codepoint_t      ab,
			   hb_codepoint_t     *a,
			   hb_codepoint_t     *b,
			   void               *user_data HB_UNUSED)
{
#if GLIB_CHECK_VERSION(2,29,12)
  return g_unichar_decompose (ab, a, b);
#endif

  /* We don't ifdef-out the fallback code such that compiler always
   * sees it and makes sure it's compilable. */

  gchar utf8[6];
  gchar *normalized;
  int len;
  hb_bool_t ret;

  len = g_unichar_to_utf8 (ab, utf8);
  normalized = g_utf8_normalize (utf8, len, G_NORMALIZE_NFD);
  len = g_utf8_strlen (normalized, -1);
  if (unlikely (!len))
    return false;

  if (len == 1) {
    *a = g_utf8_get_char (normalized);
    *b = 0;
    ret = *a != ab;
  } else if (len == 2) {
    *a = g_utf8_get_char (normalized);
    *b = g_utf8_get_char (g_utf8_next_char (normalized));
    /* Here's the ugly part: if ab decomposes to a single character and
     * that character decomposes again, we have to detect that and undo
     * the second part :-(. */
    gchar *recomposed = g_utf8_normalize (normalized, -1, G_NORMALIZE_NFC);
    hb_codepoint_t c = g_utf8_get_char (recomposed);
    if (c != ab && c != *a) {
      *a = c;
      *b = 0;
    }
    g_free (recomposed);
    ret = true;
  } else {
    /* If decomposed to more than two characters, take the last one,
     * and recompose the rest to get the first component. */
    gchar *end = g_utf8_offset_to_pointer (normalized, len - 1);
    gchar *recomposed;
    *b = g_utf8_get_char (end);
    recomposed = g_utf8_normalize (normalized, end - normalized, G_NORMALIZE_NFC);
    /* We expect that recomposed has exactly one character now. */
    *a = g_utf8_get_char (recomposed);
    g_free (recomposed);
    ret = true;
  }

  g_free (normalized);
  return ret;
}


#if HB_USE_ATEXIT
static void free_static_glib_funcs ();
#endif

static struct hb_glib_unicode_funcs_lazy_loader_t : hb_unicode_funcs_lazy_loader_t<hb_glib_unicode_funcs_lazy_loader_t>
{
  static hb_unicode_funcs_t *create ()
  {
    hb_unicode_funcs_t *funcs = hb_unicode_funcs_create (nullptr);

    hb_unicode_funcs_set_combining_class_func (funcs, hb_glib_unicode_combining_class, nullptr, nullptr);
    hb_unicode_funcs_set_general_category_func (funcs, hb_glib_unicode_general_category, nullptr, nullptr);
    hb_unicode_funcs_set_mirroring_func (funcs, hb_glib_unicode_mirroring, nullptr, nullptr);
    hb_unicode_funcs_set_script_func (funcs, hb_glib_unicode_script, nullptr, nullptr);
    hb_unicode_funcs_set_compose_func (funcs, hb_glib_unicode_compose, nullptr, nullptr);
    hb_unicode_funcs_set_decompose_func (funcs, hb_glib_unicode_decompose, nullptr, nullptr);

    hb_unicode_funcs_make_immutable (funcs);

#if HB_USE_ATEXIT
    atexit (free_static_glib_funcs);
#endif

    return funcs;
  }
} static_glib_funcs;

#if HB_USE_ATEXIT
static
void free_static_glib_funcs ()
{
  static_glib_funcs.free_instance ();
}
#endif

/**
 * hb_glib_get_unicode_funcs:
 *
 * Fetches a Unicode-functions structure that is populated
 * with the appropriate GLib function for each method.
 *
 * Return value: (transfer none): a pointer to the #hb_unicode_funcs_t Unicode-functions structure
 *
 * Since: 0.9.38
 **/
hb_unicode_funcs_t *
hb_glib_get_unicode_funcs ()
{
  return static_glib_funcs.get_unconst ();
}



#if GLIB_CHECK_VERSION(2,31,10)

static void
_hb_g_bytes_unref (void *data)
{
  g_bytes_unref ((GBytes *) data);
}

/**
 * hb_glib_blob_create:
 * @gbytes: the GBytes structure to work upon
 *
 * Creates an #hb_blob_t blob from the specified
 * GBytes data structure.
 *
 * Return value: (transfer full): the new #hb_blob_t blob object
 *
 * Since: 0.9.38
 **/
hb_blob_t *
hb_glib_blob_create (GBytes *gbytes)
{
  gsize size = 0;
  gconstpointer data = g_bytes_get_data (gbytes, &size);
  return hb_blob_create ((const char *) data,
			 size,
			 HB_MEMORY_MODE_READONLY,
			 g_bytes_ref (gbytes),
			 _hb_g_bytes_unref);
}
#endif


#endif
