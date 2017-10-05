/*
 * Copyright © 1998-2004  David Turner and Werner Lemberg
 * Copyright © 2006  Behdad Esfahbod
 * Copyright © 2007,2008,2009  Red Hat, Inc.
 * Copyright © 2012,2013  Google, Inc.
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

#include "hb-open-type-private.hh"
#include "hb-ot-layout-private.hh"

#include "hb-ot-layout-gdef-table.hh"
#include "hb-ot-layout-gsub-table.hh"
#include "hb-ot-layout-gpos-table.hh"
#include "hb-ot-layout-jstf-table.hh" // Just so we compile it; unused otherwise.
#include "hb-ot-name-table.hh" // Just so we compile it; unused otherwise.

#include "hb-ot-map-private.hh"


const void * const OT::_hb_NullPool[HB_NULL_POOL_SIZE / sizeof (void *)] = {};


hb_ot_layout_t *
_hb_ot_layout_create (hb_face_t *face)
{
  hb_ot_layout_t *layout = (hb_ot_layout_t *) calloc (1, sizeof (hb_ot_layout_t));
  if (unlikely (!layout))
    return nullptr;

  layout->gdef_blob = OT::Sanitizer<OT::GDEF>().sanitize (face->reference_table (HB_OT_TAG_GDEF));
  layout->gdef = OT::Sanitizer<OT::GDEF>::lock_instance (layout->gdef_blob);

  layout->gsub_blob = OT::Sanitizer<OT::GSUB>().sanitize (face->reference_table (HB_OT_TAG_GSUB));
  layout->gsub = OT::Sanitizer<OT::GSUB>::lock_instance (layout->gsub_blob);

  layout->gpos_blob = OT::Sanitizer<OT::GPOS>().sanitize (face->reference_table (HB_OT_TAG_GPOS));
  layout->gpos = OT::Sanitizer<OT::GPOS>::lock_instance (layout->gpos_blob);

  layout->math.init (face);
  layout->fvar.init (face);
  layout->avar.init (face);
  layout->morx.init (face);

  {
    /*
     * The ugly business of blacklisting individual fonts' tables happen here!
     * See this thread for why we finally had to bend in and do this:
     * https://lists.freedesktop.org/archives/harfbuzz/2016-February/005489.html
     */
    unsigned int gdef_len = hb_blob_get_length (layout->gdef_blob);
    unsigned int gsub_len = hb_blob_get_length (layout->gsub_blob);
    unsigned int gpos_len = hb_blob_get_length (layout->gpos_blob);
    if (0
      /* sha1sum:c5ee92f0bca4bfb7d06c4d03e8cf9f9cf75d2e8a Windows 7? timesi.ttf */
      || (442 == gdef_len && 42038 == gpos_len && 2874 == gsub_len)
      /* sha1sum:37fc8c16a0894ab7b749e35579856c73c840867b Windows 7? timesbi.ttf */
      || (430 == gdef_len && 40662 == gpos_len && 2874 == gsub_len)
      /* sha1sum:19fc45110ea6cd3cdd0a5faca256a3797a069a80 Windows 7 timesi.ttf */
      || (442 == gdef_len && 39116 == gpos_len && 2874 == gsub_len)
      /* sha1sum:6d2d3c9ed5b7de87bc84eae0df95ee5232ecde26 Windows 7 timesbi.ttf */
      || (430 == gdef_len && 39374 == gpos_len && 2874 == gsub_len)
      /* sha1sum:8583225a8b49667c077b3525333f84af08c6bcd8 OS X 10.11.3 Times New Roman Italic.ttf */
      || (490 == gdef_len && 41638 == gpos_len && 3046 == gsub_len)
      /* sha1sum:ec0f5a8751845355b7c3271d11f9918a966cb8c9 OS X 10.11.3 Times New Roman Bold Italic.ttf */
      || (478 == gdef_len && 41902 == gpos_len && 3046 == gsub_len)
    )
    {
      /* In certain versions of Times New Roman Italic and Bold Italic,
       * ASCII double quotation mark U+0022, mapped to glyph 5, has wrong
       * glyph class 3 (mark) in GDEF.  Nuke the GDEF to avoid zero-width
       * double-quote.  See:
       * https://lists.freedesktop.org/archives/harfbuzz/2016-February/005489.html
       */
     if (3 == layout->gdef->get_glyph_class (5))
       layout->gdef = &OT::Null(OT::GDEF);
    }
    else if (0
      /* sha1sum:96eda93f7d33e79962451c6c39a6b51ee893ce8c  tahoma.ttf from Windows 8 */
      || (898 == gdef_len && 46470 == gpos_len && 12554 == gsub_len)
      /* sha1sum:20928dc06014e0cd120b6fc942d0c3b1a46ac2bc  tahomabd.ttf from Windows 8 */
      || (910 == gdef_len && 47732 == gpos_len && 12566 == gsub_len)
      /* sha1sum:4f95b7e4878f60fa3a39ca269618dfde9721a79e  tahoma.ttf from Windows 8.1 */
      || (928 == gdef_len && 59332 == gpos_len && 23298 == gsub_len)
      /* sha1sum:6d400781948517c3c0441ba42acb309584b73033  tahomabd.ttf from Windows 8.1 */
      || (940 == gdef_len && 60732 == gpos_len && 23310 == gsub_len)
      /* tahoma.ttf v6.04 from Windows 8.1 x64, see https://bugzilla.mozilla.org/show_bug.cgi?id=1279925 */
      || (964 == gdef_len && 60072 == gpos_len && 23836 == gsub_len)
      /* tahomabd.ttf v6.04 from Windows 8.1 x64, see https://bugzilla.mozilla.org/show_bug.cgi?id=1279925 */
      || (976 == gdef_len && 61456 == gpos_len && 23832 == gsub_len)
      /* sha1sum:e55fa2dfe957a9f7ec26be516a0e30b0c925f846  tahoma.ttf from Windows 10 */
      || (994 == gdef_len && 60336 == gpos_len && 24474 == gsub_len)
      /* sha1sum:7199385abb4c2cc81c83a151a7599b6368e92343  tahomabd.ttf from Windows 10 */
      || (1006 == gdef_len && 61740 == gpos_len && 24470 == gsub_len)
      /* tahoma.ttf v6.91 from Windows 10 x64, see https://bugzilla.mozilla.org/show_bug.cgi?id=1279925 */
      || (1006 == gdef_len && 61346 == gpos_len && 24576 == gsub_len)
      /* tahomabd.ttf v6.91 from Windows 10 x64, see https://bugzilla.mozilla.org/show_bug.cgi?id=1279925 */
      || (1018 == gdef_len && 62828 == gpos_len && 24572 == gsub_len)
      /* sha1sum:b9c84d820c49850d3d27ec498be93955b82772b5  tahoma.ttf from Windows 10 AU */
      || (1006 == gdef_len && 61352 == gpos_len && 24576 == gsub_len)
      /* sha1sum:2bdfaab28174bdadd2f3d4200a30a7ae31db79d2  tahomabd.ttf from Windows 10 AU */
      || (1018 == gdef_len && 62834 == gpos_len && 24572 == gsub_len)
      /* sha1sum:b0d36cf5a2fbe746a3dd277bffc6756a820807a7  Tahoma.ttf from Mac OS X 10.9 */
      || (832 == gdef_len && 47162 == gpos_len && 7324 == gsub_len)
      /* sha1sum:12fc4538e84d461771b30c18b5eb6bd434e30fba  Tahoma Bold.ttf from Mac OS X 10.9 */
      || (844 == gdef_len && 45474 == gpos_len && 7302 == gsub_len)
      /* sha1sum:eb8afadd28e9cf963e886b23a30b44ab4fd83acc  himalaya.ttf from Windows 7 */
      || (180 == gdef_len && 7254 == gpos_len && 13054 == gsub_len)
      /* sha1sum:73da7f025b238a3f737aa1fde22577a6370f77b0  himalaya.ttf from Windows 8 */
      || (192 == gdef_len && 7254 == gpos_len && 12638 == gsub_len)
      /* sha1sum:6e80fd1c0b059bbee49272401583160dc1e6a427  himalaya.ttf from Windows 8.1 */
      || (192 == gdef_len && 7254 == gpos_len && 12690 == gsub_len)
      /* 8d9267aea9cd2c852ecfb9f12a6e834bfaeafe44  cantarell-fonts-0.0.21/otf/Cantarell-Regular.otf */
      /* 983988ff7b47439ab79aeaf9a45bd4a2c5b9d371  cantarell-fonts-0.0.21/otf/Cantarell-Oblique.otf */
      || (188 == gdef_len && 3852 == gpos_len && 248 == gsub_len)
      /* 2c0c90c6f6087ffbfea76589c93113a9cbb0e75f  cantarell-fonts-0.0.21/otf/Cantarell-Bold.otf */
      /* 55461f5b853c6da88069ffcdf7f4dd3f8d7e3e6b  cantarell-fonts-0.0.21/otf/Cantarell-Bold-Oblique.otf */
      || (188 == gdef_len && 3426 == gpos_len && 264 == gsub_len)
      /* d125afa82a77a6475ac0e74e7c207914af84b37a padauk-2.80/Padauk.ttf RHEL 7.2 */
      || (1058 == gdef_len && 11818 == gpos_len && 47032 == gsub_len)
      /* 0f7b80437227b90a577cc078c0216160ae61b031 padauk-2.80/Padauk-Bold.ttf RHEL 7.2*/
      || (1046 == gdef_len && 12600 == gpos_len && 47030 == gsub_len)
      /* d3dde9aa0a6b7f8f6a89ef1002e9aaa11b882290 padauk-2.80/Padauk.ttf Ubuntu 16.04 */
      || (1058 == gdef_len && 16770 == gpos_len && 71796 == gsub_len)
      /* 5f3c98ccccae8a953be2d122c1b3a77fd805093f padauk-2.80/Padauk-Bold.ttf Ubuntu 16.04 */
      || (1046 == gdef_len && 17862 == gpos_len && 71790 == gsub_len)
      /* 6c93b63b64e8b2c93f5e824e78caca555dc887c7 padauk-2.80/Padauk-book.ttf */
      || (1046 == gdef_len && 17112 == gpos_len && 71788 == gsub_len)
      /* d89b1664058359b8ec82e35d3531931125991fb9 padauk-2.80/Padauk-bookbold.ttf */
      || (1058 == gdef_len && 17514 == gpos_len && 71794 == gsub_len)
      /* 824cfd193aaf6234b2b4dc0cf3c6ef576c0d00ef padauk-3.0/Padauk-book.ttf */
      || (1330 == gdef_len && 57938 == gpos_len && 109904 == gsub_len)
      /* 91fcc10cf15e012d27571e075b3b4dfe31754a8a padauk-3.0/Padauk-bookbold.ttf */
      || (1330 == gdef_len && 58972 == gpos_len && 109904 == gsub_len)
      /* sha1sum: c26e41d567ed821bed997e937bc0c41435689e85  Padauk.ttf
       *  "Padauk Regular" "Version 2.5", see https://crbug.com/681813 */
      || (1004 == gdef_len && 14836 == gpos_len && 59092 == gsub_len)
    )
    {
      /* Many versions of Tahoma have bad GDEF tables that incorrectly classify some spacing marks
       * such as certain IPA symbols as glyph class 3. So do older versions of Microsoft Himalaya,
       * and the version of Cantarell shipped by Ubuntu 16.04.
       * Nuke the GDEF tables of these fonts to avoid unwanted width-zeroing.
       * See https://bugzilla.mozilla.org/show_bug.cgi?id=1279925
       *     https://bugzilla.mozilla.org/show_bug.cgi?id=1279693
       *     https://bugzilla.mozilla.org/show_bug.cgi?id=1279875
       */
      layout->gdef = &OT::Null(OT::GDEF);
    }
  }

  layout->gsub_lookup_count = layout->gsub->get_lookup_count ();
  layout->gpos_lookup_count = layout->gpos->get_lookup_count ();

  layout->gsub_accels = (hb_ot_layout_lookup_accelerator_t *) calloc (layout->gsub->get_lookup_count (), sizeof (hb_ot_layout_lookup_accelerator_t));
  layout->gpos_accels = (hb_ot_layout_lookup_accelerator_t *) calloc (layout->gpos->get_lookup_count (), sizeof (hb_ot_layout_lookup_accelerator_t));

  if (unlikely ((layout->gsub_lookup_count && !layout->gsub_accels) ||
		(layout->gpos_lookup_count && !layout->gpos_accels)))
  {
    _hb_ot_layout_destroy (layout);
    return nullptr;
  }

  for (unsigned int i = 0; i < layout->gsub_lookup_count; i++)
    layout->gsub_accels[i].init (layout->gsub->get_lookup (i));
  for (unsigned int i = 0; i < layout->gpos_lookup_count; i++)
    layout->gpos_accels[i].init (layout->gpos->get_lookup (i));

  return layout;
}

void
_hb_ot_layout_destroy (hb_ot_layout_t *layout)
{
  for (unsigned int i = 0; i < layout->gsub_lookup_count; i++)
    layout->gsub_accels[i].fini ();
  for (unsigned int i = 0; i < layout->gpos_lookup_count; i++)
    layout->gpos_accels[i].fini ();

  free (layout->gsub_accels);
  free (layout->gpos_accels);

  hb_blob_destroy (layout->gdef_blob);
  hb_blob_destroy (layout->gsub_blob);
  hb_blob_destroy (layout->gpos_blob);

  layout->math.fini ();
  layout->fvar.fini ();
  layout->avar.fini ();
  layout->morx.fini ();

  free (layout);
}

static inline const OT::GDEF&
_get_gdef (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::GDEF);
  return *hb_ot_layout_from_face (face)->gdef;
}
static inline const OT::GSUB&
_get_gsub (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::GSUB);
  return *hb_ot_layout_from_face (face)->gsub;
}
static inline const OT::GPOS&
_get_gpos (hb_face_t *face)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return OT::Null(OT::GPOS);
  return *hb_ot_layout_from_face (face)->gpos;
}

/*
 * GDEF
 */

hb_bool_t
hb_ot_layout_has_glyph_classes (hb_face_t *face)
{
  return _get_gdef (face).has_glyph_classes ();
}

/**
 * hb_ot_layout_get_glyph_class:
 *
 * Since: 0.9.7
 **/
hb_ot_layout_glyph_class_t
hb_ot_layout_get_glyph_class (hb_face_t      *face,
			      hb_codepoint_t  glyph)
{
  return (hb_ot_layout_glyph_class_t) _get_gdef (face).get_glyph_class (glyph);
}

/**
 * hb_ot_layout_get_glyphs_in_class:
 *
 * Since: 0.9.7
 **/
void
hb_ot_layout_get_glyphs_in_class (hb_face_t                  *face,
				  hb_ot_layout_glyph_class_t  klass,
				  hb_set_t                   *glyphs /* OUT */)
{
  return _get_gdef (face).get_glyphs_in_class (klass, glyphs);
}

unsigned int
hb_ot_layout_get_attach_points (hb_face_t      *face,
				hb_codepoint_t  glyph,
				unsigned int    start_offset,
				unsigned int   *point_count /* IN/OUT */,
				unsigned int   *point_array /* OUT */)
{
  return _get_gdef (face).get_attach_points (glyph, start_offset, point_count, point_array);
}

unsigned int
hb_ot_layout_get_ligature_carets (hb_font_t      *font,
				  hb_direction_t  direction,
				  hb_codepoint_t  glyph,
				  unsigned int    start_offset,
				  unsigned int   *caret_count /* IN/OUT */,
				  hb_position_t  *caret_array /* OUT */)
{
  return _get_gdef (font->face).get_lig_carets (font, direction, glyph, start_offset, caret_count, caret_array);
}


/*
 * GSUB/GPOS
 */

static const OT::GSUBGPOS&
get_gsubgpos_table (hb_face_t *face,
		    hb_tag_t   table_tag)
{
  switch (table_tag) {
    case HB_OT_TAG_GSUB: return _get_gsub (face);
    case HB_OT_TAG_GPOS: return _get_gpos (face);
    default:             return OT::Null(OT::GSUBGPOS);
  }
}


unsigned int
hb_ot_layout_table_get_script_tags (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  start_offset,
				    unsigned int *script_count /* IN/OUT */,
				    hb_tag_t     *script_tags /* OUT */)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  return g.get_script_tags (start_offset, script_count, script_tags);
}

#define HB_OT_TAG_LATIN_SCRIPT		HB_TAG ('l', 'a', 't', 'n')

hb_bool_t
hb_ot_layout_table_find_script (hb_face_t    *face,
				hb_tag_t      table_tag,
				hb_tag_t      script_tag,
				unsigned int *script_index)
{
  static_assert ((OT::Index::NOT_FOUND_INDEX == HB_OT_LAYOUT_NO_SCRIPT_INDEX), "");
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  if (g.find_script_index (script_tag, script_index))
    return true;

  /* try finding 'DFLT' */
  if (g.find_script_index (HB_OT_TAG_DEFAULT_SCRIPT, script_index))
    return false;

  /* try with 'dflt'; MS site has had typos and many fonts use it now :(.
   * including many versions of DejaVu Sans Mono! */
  if (g.find_script_index (HB_OT_TAG_DEFAULT_LANGUAGE, script_index))
    return false;

  /* try with 'latn'; some old fonts put their features there even though
     they're really trying to support Thai, for example :( */
  if (g.find_script_index (HB_OT_TAG_LATIN_SCRIPT, script_index))
    return false;

  if (script_index) *script_index = HB_OT_LAYOUT_NO_SCRIPT_INDEX;
  return false;
}

hb_bool_t
hb_ot_layout_table_choose_script (hb_face_t      *face,
				  hb_tag_t        table_tag,
				  const hb_tag_t *script_tags,
				  unsigned int   *script_index,
				  hb_tag_t       *chosen_script)
{
  static_assert ((OT::Index::NOT_FOUND_INDEX == HB_OT_LAYOUT_NO_SCRIPT_INDEX), "");
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  while (*script_tags)
  {
    if (g.find_script_index (*script_tags, script_index)) {
      if (chosen_script)
        *chosen_script = *script_tags;
      return true;
    }
    script_tags++;
  }

  /* try finding 'DFLT' */
  if (g.find_script_index (HB_OT_TAG_DEFAULT_SCRIPT, script_index)) {
    if (chosen_script)
      *chosen_script = HB_OT_TAG_DEFAULT_SCRIPT;
    return false;
  }

  /* try with 'dflt'; MS site has had typos and many fonts use it now :( */
  if (g.find_script_index (HB_OT_TAG_DEFAULT_LANGUAGE, script_index)) {
    if (chosen_script)
      *chosen_script = HB_OT_TAG_DEFAULT_LANGUAGE;
    return false;
  }

  /* try with 'latn'; some old fonts put their features there even though
     they're really trying to support Thai, for example :( */
  if (g.find_script_index (HB_OT_TAG_LATIN_SCRIPT, script_index)) {
    if (chosen_script)
      *chosen_script = HB_OT_TAG_LATIN_SCRIPT;
    return false;
  }

  if (script_index) *script_index = HB_OT_LAYOUT_NO_SCRIPT_INDEX;
  if (chosen_script)
    *chosen_script = HB_OT_LAYOUT_NO_SCRIPT_INDEX;
  return false;
}

unsigned int
hb_ot_layout_table_get_feature_tags (hb_face_t    *face,
				     hb_tag_t      table_tag,
				     unsigned int  start_offset,
				     unsigned int *feature_count /* IN/OUT */,
				     hb_tag_t     *feature_tags /* OUT */)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  return g.get_feature_tags (start_offset, feature_count, feature_tags);
}

hb_bool_t
hb_ot_layout_table_find_feature (hb_face_t    *face,
				 hb_tag_t      table_tag,
				 hb_tag_t      feature_tag,
				 unsigned int *feature_index)
{
  static_assert ((OT::Index::NOT_FOUND_INDEX == HB_OT_LAYOUT_NO_FEATURE_INDEX), "");
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  unsigned int num_features = g.get_feature_count ();
  for (unsigned int i = 0; i < num_features; i++)
  {
    if (feature_tag == g.get_feature_tag (i)) {
      if (feature_index) *feature_index = i;
      return true;
    }
  }

  if (feature_index) *feature_index = HB_OT_LAYOUT_NO_FEATURE_INDEX;
  return false;
}


unsigned int
hb_ot_layout_script_get_language_tags (hb_face_t    *face,
				       hb_tag_t      table_tag,
				       unsigned int  script_index,
				       unsigned int  start_offset,
				       unsigned int *language_count /* IN/OUT */,
				       hb_tag_t     *language_tags /* OUT */)
{
  const OT::Script &s = get_gsubgpos_table (face, table_tag).get_script (script_index);

  return s.get_lang_sys_tags (start_offset, language_count, language_tags);
}

hb_bool_t
hb_ot_layout_script_find_language (hb_face_t    *face,
				   hb_tag_t      table_tag,
				   unsigned int  script_index,
				   hb_tag_t      language_tag,
				   unsigned int *language_index)
{
  static_assert ((OT::Index::NOT_FOUND_INDEX == HB_OT_LAYOUT_DEFAULT_LANGUAGE_INDEX), "");
  const OT::Script &s = get_gsubgpos_table (face, table_tag).get_script (script_index);

  if (s.find_lang_sys_index (language_tag, language_index))
    return true;

  /* try with 'dflt'; MS site has had typos and many fonts use it now :( */
  if (s.find_lang_sys_index (HB_OT_TAG_DEFAULT_LANGUAGE, language_index))
    return false;

  if (language_index) *language_index = HB_OT_LAYOUT_DEFAULT_LANGUAGE_INDEX;
  return false;
}

hb_bool_t
hb_ot_layout_language_get_required_feature_index (hb_face_t    *face,
						  hb_tag_t      table_tag,
						  unsigned int  script_index,
						  unsigned int  language_index,
						  unsigned int *feature_index)
{
  return hb_ot_layout_language_get_required_feature (face,
						     table_tag,
						     script_index,
						     language_index,
						     feature_index,
						     nullptr);
}

/**
 * hb_ot_layout_language_get_required_feature:
 *
 * Since: 0.9.30
 **/
hb_bool_t
hb_ot_layout_language_get_required_feature (hb_face_t    *face,
					    hb_tag_t      table_tag,
					    unsigned int  script_index,
					    unsigned int  language_index,
					    unsigned int *feature_index,
					    hb_tag_t     *feature_tag)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);
  const OT::LangSys &l = g.get_script (script_index).get_lang_sys (language_index);

  unsigned int index = l.get_required_feature_index ();
  if (feature_index) *feature_index = index;
  if (feature_tag) *feature_tag = g.get_feature_tag (index);

  return l.has_required_feature ();
}

unsigned int
hb_ot_layout_language_get_feature_indexes (hb_face_t    *face,
					   hb_tag_t      table_tag,
					   unsigned int  script_index,
					   unsigned int  language_index,
					   unsigned int  start_offset,
					   unsigned int *feature_count /* IN/OUT */,
					   unsigned int *feature_indexes /* OUT */)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);
  const OT::LangSys &l = g.get_script (script_index).get_lang_sys (language_index);

  return l.get_feature_indexes (start_offset, feature_count, feature_indexes);
}

unsigned int
hb_ot_layout_language_get_feature_tags (hb_face_t    *face,
					hb_tag_t      table_tag,
					unsigned int  script_index,
					unsigned int  language_index,
					unsigned int  start_offset,
					unsigned int *feature_count /* IN/OUT */,
					hb_tag_t     *feature_tags /* OUT */)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);
  const OT::LangSys &l = g.get_script (script_index).get_lang_sys (language_index);

  static_assert ((sizeof (unsigned int) == sizeof (hb_tag_t)), "");
  unsigned int ret = l.get_feature_indexes (start_offset, feature_count, (unsigned int *) feature_tags);

  if (feature_tags) {
    unsigned int count = *feature_count;
    for (unsigned int i = 0; i < count; i++)
      feature_tags[i] = g.get_feature_tag ((unsigned int) feature_tags[i]);
  }

  return ret;
}


hb_bool_t
hb_ot_layout_language_find_feature (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  script_index,
				    unsigned int  language_index,
				    hb_tag_t      feature_tag,
				    unsigned int *feature_index)
{
  static_assert ((OT::Index::NOT_FOUND_INDEX == HB_OT_LAYOUT_NO_FEATURE_INDEX), "");
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);
  const OT::LangSys &l = g.get_script (script_index).get_lang_sys (language_index);

  unsigned int num_features = l.get_feature_count ();
  for (unsigned int i = 0; i < num_features; i++) {
    unsigned int f_index = l.get_feature_index (i);

    if (feature_tag == g.get_feature_tag (f_index)) {
      if (feature_index) *feature_index = f_index;
      return true;
    }
  }

  if (feature_index) *feature_index = HB_OT_LAYOUT_NO_FEATURE_INDEX;
  return false;
}

/**
 * hb_ot_layout_feature_get_lookups:
 *
 * Since: 0.9.7
 **/
unsigned int
hb_ot_layout_feature_get_lookups (hb_face_t    *face,
				  hb_tag_t      table_tag,
				  unsigned int  feature_index,
				  unsigned int  start_offset,
				  unsigned int *lookup_count /* IN/OUT */,
				  unsigned int *lookup_indexes /* OUT */)
{
  return hb_ot_layout_feature_with_variations_get_lookups (face,
							   table_tag,
							   feature_index,
							   HB_OT_LAYOUT_NO_VARIATIONS_INDEX,
							   start_offset,
							   lookup_count,
							   lookup_indexes);
}

/**
 * hb_ot_layout_table_get_lookup_count:
 *
 * Since: 0.9.22
 **/
unsigned int
hb_ot_layout_table_get_lookup_count (hb_face_t    *face,
				     hb_tag_t      table_tag)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return 0;
  switch (table_tag)
  {
    case HB_OT_TAG_GSUB:
    {
      return hb_ot_layout_from_face (face)->gsub_lookup_count;
    }
    case HB_OT_TAG_GPOS:
    {
      return hb_ot_layout_from_face (face)->gpos_lookup_count;
    }
  }
  return 0;
}

static void
_hb_ot_layout_collect_lookups_lookups (hb_face_t      *face,
				       hb_tag_t        table_tag,
				       unsigned int    feature_index,
				       hb_set_t       *lookup_indexes /* OUT */)
{
  unsigned int lookup_indices[32];
  unsigned int offset, len;

  offset = 0;
  do {
    len = ARRAY_LENGTH (lookup_indices);
    hb_ot_layout_feature_get_lookups (face,
				      table_tag,
				      feature_index,
				      offset, &len,
				      lookup_indices);

    for (unsigned int i = 0; i < len; i++)
      lookup_indexes->add (lookup_indices[i]);

    offset += len;
  } while (len == ARRAY_LENGTH (lookup_indices));
}

static void
_hb_ot_layout_collect_lookups_features (hb_face_t      *face,
					hb_tag_t        table_tag,
					unsigned int    script_index,
					unsigned int    language_index,
					const hb_tag_t *features,
					hb_set_t       *lookup_indexes /* OUT */)
{
  if (!features)
  {
    unsigned int required_feature_index;
    if (hb_ot_layout_language_get_required_feature (face,
						    table_tag,
						    script_index,
						    language_index,
						    &required_feature_index,
						    nullptr))
      _hb_ot_layout_collect_lookups_lookups (face,
					     table_tag,
					     required_feature_index,
					     lookup_indexes);

    /* All features */
    unsigned int feature_indices[32];
    unsigned int offset, len;

    offset = 0;
    do {
      len = ARRAY_LENGTH (feature_indices);
      hb_ot_layout_language_get_feature_indexes (face,
						 table_tag,
						 script_index,
						 language_index,
						 offset, &len,
						 feature_indices);

      for (unsigned int i = 0; i < len; i++)
	_hb_ot_layout_collect_lookups_lookups (face,
					       table_tag,
					       feature_indices[i],
					       lookup_indexes);

      offset += len;
    } while (len == ARRAY_LENGTH (feature_indices));
  }
  else
  {
    for (; *features; features++)
    {
      unsigned int feature_index;
      if (hb_ot_layout_language_find_feature (face,
					      table_tag,
					      script_index,
					      language_index,
					      *features,
					      &feature_index))
        _hb_ot_layout_collect_lookups_lookups (face,
					       table_tag,
					       feature_index,
					       lookup_indexes);
    }
  }
}

static void
_hb_ot_layout_collect_lookups_languages (hb_face_t      *face,
					 hb_tag_t        table_tag,
					 unsigned int    script_index,
					 const hb_tag_t *languages,
					 const hb_tag_t *features,
					 hb_set_t       *lookup_indexes /* OUT */)
{
  _hb_ot_layout_collect_lookups_features (face,
					  table_tag,
					  script_index,
					  HB_OT_LAYOUT_DEFAULT_LANGUAGE_INDEX,
					  features,
					  lookup_indexes);

  if (!languages)
  {
    /* All languages */
    unsigned int count = hb_ot_layout_script_get_language_tags (face,
								table_tag,
								script_index,
								0, nullptr, nullptr);
    for (unsigned int language_index = 0; language_index < count; language_index++)
      _hb_ot_layout_collect_lookups_features (face,
					      table_tag,
					      script_index,
					      language_index,
					      features,
					      lookup_indexes);
  }
  else
  {
    for (; *languages; languages++)
    {
      unsigned int language_index;
      if (hb_ot_layout_script_find_language (face,
					     table_tag,
					     script_index,
					     *languages,
					     &language_index))
        _hb_ot_layout_collect_lookups_features (face,
						table_tag,
						script_index,
						language_index,
						features,
						lookup_indexes);
    }
  }
}

/**
 * hb_ot_layout_collect_lookups:
 *
 * Since: 0.9.8
 **/
void
hb_ot_layout_collect_lookups (hb_face_t      *face,
			      hb_tag_t        table_tag,
			      const hb_tag_t *scripts,
			      const hb_tag_t *languages,
			      const hb_tag_t *features,
			      hb_set_t       *lookup_indexes /* OUT */)
{
  if (!scripts)
  {
    /* All scripts */
    unsigned int count = hb_ot_layout_table_get_script_tags (face,
							     table_tag,
							     0, nullptr, nullptr);
    for (unsigned int script_index = 0; script_index < count; script_index++)
      _hb_ot_layout_collect_lookups_languages (face,
					       table_tag,
					       script_index,
					       languages,
					       features,
					       lookup_indexes);
  }
  else
  {
    for (; *scripts; scripts++)
    {
      unsigned int script_index;
      if (hb_ot_layout_table_find_script (face,
					  table_tag,
					  *scripts,
					  &script_index))
        _hb_ot_layout_collect_lookups_languages (face,
						 table_tag,
						 script_index,
						 languages,
						 features,
						 lookup_indexes);
    }
  }
}

/**
 * hb_ot_layout_lookup_collect_glyphs:
 *
 * Since: 0.9.7
 **/
void
hb_ot_layout_lookup_collect_glyphs (hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  lookup_index,
				    hb_set_t     *glyphs_before, /* OUT. May be nullptr */
				    hb_set_t     *glyphs_input,  /* OUT. May be nullptr */
				    hb_set_t     *glyphs_after,  /* OUT. May be nullptr */
				    hb_set_t     *glyphs_output  /* OUT. May be nullptr */)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return;

  OT::hb_collect_glyphs_context_t c (face,
				     glyphs_before,
				     glyphs_input,
				     glyphs_after,
				     glyphs_output);

  switch (table_tag)
  {
    case HB_OT_TAG_GSUB:
    {
      const OT::SubstLookup& l = hb_ot_layout_from_face (face)->gsub->get_lookup (lookup_index);
      l.collect_glyphs (&c);
      return;
    }
    case HB_OT_TAG_GPOS:
    {
      const OT::PosLookup& l = hb_ot_layout_from_face (face)->gpos->get_lookup (lookup_index);
      l.collect_glyphs (&c);
      return;
    }
  }
}


/* Variations support */

hb_bool_t
hb_ot_layout_table_find_feature_variations (hb_face_t    *face,
					    hb_tag_t      table_tag,
					    const int    *coords,
					    unsigned int  num_coords,
					    unsigned int *variations_index /* out */)
{
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  return g.find_variations_index (coords, num_coords, variations_index);
}

unsigned int
hb_ot_layout_feature_with_variations_get_lookups (hb_face_t    *face,
						  hb_tag_t      table_tag,
						  unsigned int  feature_index,
						  unsigned int  variations_index,
						  unsigned int  start_offset,
						  unsigned int *lookup_count /* IN/OUT */,
						  unsigned int *lookup_indexes /* OUT */)
{
  static_assert ((OT::FeatureVariations::NOT_FOUND_INDEX == HB_OT_LAYOUT_NO_VARIATIONS_INDEX), "");
  const OT::GSUBGPOS &g = get_gsubgpos_table (face, table_tag);

  const OT::Feature &f = g.get_feature_variation (feature_index, variations_index);

  return f.get_lookup_indexes (start_offset, lookup_count, lookup_indexes);
}


/*
 * OT::GSUB
 */

hb_bool_t
hb_ot_layout_has_substitution (hb_face_t *face)
{
  return &_get_gsub (face) != &OT::Null(OT::GSUB);
}

/**
 * hb_ot_layout_lookup_would_substitute:
 *
 * Since: 0.9.7
 **/
hb_bool_t
hb_ot_layout_lookup_would_substitute (hb_face_t            *face,
				      unsigned int          lookup_index,
				      const hb_codepoint_t *glyphs,
				      unsigned int          glyphs_length,
				      hb_bool_t             zero_context)
{
  if (unlikely (!hb_ot_shaper_face_data_ensure (face))) return false;
  return hb_ot_layout_lookup_would_substitute_fast (face, lookup_index, glyphs, glyphs_length, zero_context);
}

hb_bool_t
hb_ot_layout_lookup_would_substitute_fast (hb_face_t            *face,
					   unsigned int          lookup_index,
					   const hb_codepoint_t *glyphs,
					   unsigned int          glyphs_length,
					   hb_bool_t             zero_context)
{
  if (unlikely (lookup_index >= hb_ot_layout_from_face (face)->gsub_lookup_count)) return false;
  OT::hb_would_apply_context_t c (face, glyphs, glyphs_length, (bool) zero_context);

  const OT::SubstLookup& l = hb_ot_layout_from_face (face)->gsub->get_lookup (lookup_index);

  return l.would_apply (&c, &hb_ot_layout_from_face (face)->gsub_accels[lookup_index]);
}

void
hb_ot_layout_substitute_start (hb_font_t *font, hb_buffer_t *buffer)
{
  OT::GSUB::substitute_start (font, buffer);
}

/**
 * hb_ot_layout_lookup_substitute_closure:
 *
 * Since: 0.9.7
 **/
void
hb_ot_layout_lookup_substitute_closure (hb_face_t    *face,
				        unsigned int  lookup_index,
				        hb_set_t     *glyphs)
{
  OT::hb_closure_context_t c (face, glyphs);

  const OT::SubstLookup& l = _get_gsub (face).get_lookup (lookup_index);

  l.closure (&c);
}

/*
 * OT::GPOS
 */

hb_bool_t
hb_ot_layout_has_positioning (hb_face_t *face)
{
  return &_get_gpos (face) != &OT::Null(OT::GPOS);
}

void
hb_ot_layout_position_start (hb_font_t *font, hb_buffer_t *buffer)
{
  OT::GPOS::position_start (font, buffer);
}

void
hb_ot_layout_position_finish_advances (hb_font_t *font, hb_buffer_t *buffer)
{
  OT::GPOS::position_finish_advances (font, buffer);
}

void
hb_ot_layout_position_finish_offsets (hb_font_t *font, hb_buffer_t *buffer)
{
  OT::GPOS::position_finish_offsets (font, buffer);
}

/**
 * hb_ot_layout_get_size_params:
 *
 * Since: 0.9.10
 **/
hb_bool_t
hb_ot_layout_get_size_params (hb_face_t    *face,
			      unsigned int *design_size,       /* OUT.  May be nullptr */
			      unsigned int *subfamily_id,      /* OUT.  May be nullptr */
			      unsigned int *subfamily_name_id, /* OUT.  May be nullptr */
			      unsigned int *range_start,       /* OUT.  May be nullptr */
			      unsigned int *range_end          /* OUT.  May be nullptr */)
{
  const OT::GPOS &gpos = _get_gpos (face);
  const hb_tag_t tag = HB_TAG ('s','i','z','e');

  unsigned int num_features = gpos.get_feature_count ();
  for (unsigned int i = 0; i < num_features; i++)
  {
    if (tag == gpos.get_feature_tag (i))
    {
      const OT::Feature &f = gpos.get_feature (i);
      const OT::FeatureParamsSize &params = f.get_feature_params ().get_size_params (tag);

      if (params.designSize)
      {
#define PARAM(a, A) if (a) *a = params.A
	PARAM (design_size, designSize);
	PARAM (subfamily_id, subfamilyID);
	PARAM (subfamily_name_id, subfamilyNameID);
	PARAM (range_start, rangeStart);
	PARAM (range_end, rangeEnd);
#undef PARAM

	return true;
      }
    }
  }

#define PARAM(a, A) if (a) *a = 0
  PARAM (design_size, designSize);
  PARAM (subfamily_id, subfamilyID);
  PARAM (subfamily_name_id, subfamilyNameID);
  PARAM (range_start, rangeStart);
  PARAM (range_end, rangeEnd);
#undef PARAM

  return false;
}


/*
 * Parts of different types are implemented here such that they have direct
 * access to GSUB/GPOS lookups.
 */


struct GSUBProxy
{
  static const unsigned int table_index = 0;
  static const bool inplace = false;
  typedef OT::SubstLookup Lookup;

  GSUBProxy (hb_face_t *face) :
    table (*hb_ot_layout_from_face (face)->gsub),
    accels (hb_ot_layout_from_face (face)->gsub_accels) {}

  const OT::GSUB &table;
  const hb_ot_layout_lookup_accelerator_t *accels;
};

struct GPOSProxy
{
  static const unsigned int table_index = 1;
  static const bool inplace = true;
  typedef OT::PosLookup Lookup;

  GPOSProxy (hb_face_t *face) :
    table (*hb_ot_layout_from_face (face)->gpos),
    accels (hb_ot_layout_from_face (face)->gpos_accels) {}

  const OT::GPOS &table;
  const hb_ot_layout_lookup_accelerator_t *accels;
};


struct hb_get_subtables_context_t :
       OT::hb_dispatch_context_t<hb_get_subtables_context_t, hb_void_t, HB_DEBUG_APPLY>
{
  template <typename Type>
  static inline bool apply_to (const void *obj, OT::hb_ot_apply_context_t *c)
  {
    const Type *typed_obj = (const Type *) obj;
    return typed_obj->apply (c);
  }

  typedef bool (*hb_apply_func_t) (const void *obj, OT::hb_ot_apply_context_t *c);

  struct hb_applicable_t
  {
    inline void init (const void *obj_, hb_apply_func_t apply_func_)
    {
      obj = obj_;
      apply_func = apply_func_;
    }

    inline bool apply (OT::hb_ot_apply_context_t *c) const { return apply_func (obj, c); }

    private:
    const void *obj;
    hb_apply_func_t apply_func;
  };

  typedef hb_auto_array_t<hb_applicable_t> array_t;

  /* Dispatch interface. */
  inline const char *get_name (void) { return "GET_SUBTABLES"; }
  template <typename T>
  inline return_t dispatch (const T &obj)
  {
    hb_applicable_t *entry = array.push();
    if (likely (entry))
      entry->init (&obj, apply_to<T>);
    return HB_VOID;
  }
  static return_t default_return_value (void) { return HB_VOID; }
  bool stop_sublookup_iteration (return_t r HB_UNUSED) const { return false; }

  hb_get_subtables_context_t (array_t &array_) :
			      array (array_),
			      debug_depth (0) {}

  array_t &array;
  unsigned int debug_depth;
};

static inline bool
apply_forward (OT::hb_ot_apply_context_t *c,
	       const hb_ot_layout_lookup_accelerator_t &accel,
	       const hb_get_subtables_context_t::array_t &subtables)
{
  bool ret = false;
  hb_buffer_t *buffer = c->buffer;
  while (buffer->idx < buffer->len && !buffer->in_error)
  {
    bool applied = false;
    if (accel.may_have (buffer->cur().codepoint) &&
	(buffer->cur().mask & c->lookup_mask) &&
	c->check_glyph_property (&buffer->cur(), c->lookup_props))
     {
       for (unsigned int i = 0; i < subtables.len; i++)
         if (subtables[i].apply (c))
	 {
	   applied = true;
	   break;
	 }
     }

    if (applied)
      ret = true;
    else
      buffer->next_glyph ();
  }
  return ret;
}

static inline bool
apply_backward (OT::hb_ot_apply_context_t *c,
	       const hb_ot_layout_lookup_accelerator_t &accel,
	       const hb_get_subtables_context_t::array_t &subtables)
{
  bool ret = false;
  hb_buffer_t *buffer = c->buffer;
  do
  {
    if (accel.may_have (buffer->cur().codepoint) &&
	(buffer->cur().mask & c->lookup_mask) &&
	c->check_glyph_property (&buffer->cur(), c->lookup_props))
    {
     for (unsigned int i = 0; i < subtables.len; i++)
       if (subtables[i].apply (c))
       {
	 ret = true;
	 break;
       }
    }
    /* The reverse lookup doesn't "advance" cursor (for good reason). */
    buffer->idx--;

  }
  while ((int) buffer->idx >= 0);
  return ret;
}

template <typename Proxy>
static inline void
apply_string (OT::hb_ot_apply_context_t *c,
	      const typename Proxy::Lookup &lookup,
	      const hb_ot_layout_lookup_accelerator_t &accel)
{
  hb_buffer_t *buffer = c->buffer;

  if (unlikely (!buffer->len || !c->lookup_mask))
    return;

  c->set_lookup_props (lookup.get_props ());

  hb_get_subtables_context_t::array_t subtables;
  hb_get_subtables_context_t c_get_subtables (subtables);
  lookup.dispatch (&c_get_subtables);

  if (likely (!lookup.is_reverse ()))
  {
    /* in/out forward substitution/positioning */
    if (Proxy::table_index == 0)
      buffer->clear_output ();
    buffer->idx = 0;

    bool ret;
    ret = apply_forward (c, accel, subtables);
    if (ret)
    {
      if (!Proxy::inplace)
	buffer->swap_buffers ();
      else
	assert (!buffer->has_separate_output ());
    }
  }
  else
  {
    /* in-place backward substitution/positioning */
    if (Proxy::table_index == 0)
      buffer->remove_output ();
    buffer->idx = buffer->len - 1;

    apply_backward (c, accel, subtables);
  }
}

template <typename Proxy>
inline void hb_ot_map_t::apply (const Proxy &proxy,
				const hb_ot_shape_plan_t *plan,
				hb_font_t *font,
				hb_buffer_t *buffer) const
{
  const unsigned int table_index = proxy.table_index;
  unsigned int i = 0;
  OT::hb_ot_apply_context_t c (table_index, font, buffer);
  c.set_recurse_func (Proxy::Lookup::apply_recurse_func);

  for (unsigned int stage_index = 0; stage_index < stages[table_index].len; stage_index++) {
    const stage_map_t *stage = &stages[table_index][stage_index];
    for (; i < stage->last_lookup; i++)
    {
      unsigned int lookup_index = lookups[table_index][i].index;
      if (!buffer->message (font, "start lookup %d", lookup_index)) continue;
      c.set_lookup_index (lookup_index);
      c.set_lookup_mask (lookups[table_index][i].mask);
      c.set_auto_zwj (lookups[table_index][i].auto_zwj);
      c.set_auto_zwnj (lookups[table_index][i].auto_zwnj);
      apply_string<Proxy> (&c,
			   proxy.table.get_lookup (lookup_index),
			   proxy.accels[lookup_index]);
      (void) buffer->message (font, "end lookup %d", lookup_index);
    }

    if (stage->pause_func)
    {
      buffer->clear_output ();
      stage->pause_func (plan, font, buffer);
    }
  }
}

void hb_ot_map_t::substitute (const hb_ot_shape_plan_t *plan, hb_font_t *font, hb_buffer_t *buffer) const
{
  GSUBProxy proxy (font->face);
  apply (proxy, plan, font, buffer);
}

void hb_ot_map_t::position (const hb_ot_shape_plan_t *plan, hb_font_t *font, hb_buffer_t *buffer) const
{
  GPOSProxy proxy (font->face);
  apply (proxy, plan, font, buffer);
}

HB_INTERNAL void
hb_ot_layout_substitute_lookup (OT::hb_ot_apply_context_t *c,
				const OT::SubstLookup &lookup,
				const hb_ot_layout_lookup_accelerator_t &accel)
{
  apply_string<GSUBProxy> (c, lookup, accel);
}
