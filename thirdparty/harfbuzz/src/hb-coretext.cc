/*
 * Copyright © 2012,2013  Mozilla Foundation.
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
 * Mozilla Author(s): Jonathan Kew
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifdef HAVE_CORETEXT

#include "hb-shaper-impl.hh"

#include "hb-coretext.hh"


/**
 * SECTION:hb-coretext
 * @title: hb-coretext
 * @short_description: CoreText integration
 * @include: hb-coretext.h
 *
 * Functions for using HarfBuzz with the CoreText fonts.
 **/

static void
release_table_data (void *user_data)
{
  CFDataRef cf_data = reinterpret_cast<CFDataRef> (user_data);
  CFRelease(cf_data);
}

static hb_blob_t *
_hb_cg_reference_table (hb_face_t *face HB_UNUSED, hb_tag_t tag, void *user_data)
{
  CGFontRef cg_font = reinterpret_cast<CGFontRef> (user_data);
  CFDataRef cf_data = CGFontCopyTableForTag (cg_font, tag);
  if (unlikely (!cf_data))
    return nullptr;

  const char *data = reinterpret_cast<const char*> (CFDataGetBytePtr (cf_data));
  const size_t length = CFDataGetLength (cf_data);
  if (!data || !length)
  {
    CFRelease (cf_data);
    return nullptr;
  }

  return hb_blob_create (data, length, HB_MEMORY_MODE_READONLY,
			 reinterpret_cast<void *> (const_cast<__CFData *> (cf_data)),
			 release_table_data);
}

static unsigned
_hb_cg_get_table_tags (const hb_face_t *face HB_UNUSED,
		       unsigned int start_offset,
		       unsigned int *table_count,
		       hb_tag_t *table_tags,
		       void *user_data)
{
  CGFontRef cg_font = reinterpret_cast<CGFontRef> (user_data);

  CTFontRef ct_font = create_ct_font (cg_font, (CGFloat) HB_CORETEXT_DEFAULT_FONT_SIZE);

  auto arr = CTFontCopyAvailableTables (ct_font, kCTFontTableOptionNoOptions);

  unsigned population = (unsigned) CFArrayGetCount (arr);
  unsigned end_offset;

  if (!table_count)
    goto done;

  if (unlikely (start_offset >= population))
  {
    *table_count = 0;
    goto done;
  }

  end_offset = start_offset + *table_count;
  if (unlikely (end_offset < start_offset))
  {
    *table_count = 0;
    goto done;
  }
  end_offset= hb_min (end_offset, (unsigned) population);

  *table_count = end_offset - start_offset;
  for (unsigned i = start_offset; i < end_offset; i++)
  {
    CTFontTableTag tag = (CTFontTableTag)(uintptr_t) CFArrayGetValueAtIndex (arr, i);
    table_tags[i - start_offset] = tag;
  }

done:
  CFRelease (arr);
  CFRelease (ct_font);
  return population;
}

static void
_hb_cg_font_release (void *data)
{
  CGFontRelease ((CGFontRef) data);
}


static CTFontDescriptorRef
get_last_resort_font_desc ()
{
  // TODO Handle allocation failures?
  CTFontDescriptorRef last_resort = CTFontDescriptorCreateWithNameAndSize (CFSTR("LastResort"), 0);
  CFArrayRef cascade_list = CFArrayCreate (kCFAllocatorDefault,
					   (const void **) &last_resort,
					   1,
					   &kCFTypeArrayCallBacks);
  CFRelease (last_resort);
  CFDictionaryRef attributes = CFDictionaryCreate (kCFAllocatorDefault,
						   (const void **) &kCTFontCascadeListAttribute,
						   (const void **) &cascade_list,
						   1,
						   &kCFTypeDictionaryKeyCallBacks,
						   &kCFTypeDictionaryValueCallBacks);
  CFRelease (cascade_list);

  CTFontDescriptorRef font_desc = CTFontDescriptorCreateWithAttributes (attributes);
  CFRelease (attributes);
  return font_desc;
}

static void
release_data (void *info, const void *data, size_t size)
{
  assert (hb_blob_get_length ((hb_blob_t *) info) == size &&
	  hb_blob_get_data ((hb_blob_t *) info, nullptr) == data);

  hb_blob_destroy ((hb_blob_t *) info);
}

CGFontRef
create_cg_font (CFArrayRef ct_font_desc_array, unsigned int named_instance_index)
{
  if (named_instance_index == 0)
  {
    // Default instance. We don't know which one is it. Return the first one.
    // We will set the correct variations on it later.
  }
  else
    named_instance_index--;
  auto ct_font_desc = (CFArrayGetCount (ct_font_desc_array) > named_instance_index) ?
		      (CTFontDescriptorRef) CFArrayGetValueAtIndex (ct_font_desc_array, named_instance_index) : nullptr;
  if (unlikely (!ct_font_desc))
  {
    CFRelease (ct_font_desc_array);
    return nullptr;
  }
  auto ct_font = ct_font_desc ? CTFontCreateWithFontDescriptor (ct_font_desc, 0, nullptr) : nullptr;
  CFRelease (ct_font_desc_array);
  if (unlikely (!ct_font))
    return nullptr;

  auto cg_font = ct_font ? CTFontCopyGraphicsFont (ct_font, nullptr) : nullptr;
  CFRelease (ct_font);

  return cg_font;
}

CGFontRef
create_cg_font (hb_blob_t *blob, unsigned int index)
{
  hb_blob_make_immutable (blob);
  unsigned int blob_length;
  const char *blob_data = hb_blob_get_data (blob, &blob_length);
  if (unlikely (!blob_length))
    DEBUG_MSG (CORETEXT, blob, "Empty blob");

  unsigned ttc_index = index & 0xFFFF;
  unsigned named_instance_index = index >> 16;

  if (ttc_index != 0)
  {
    DEBUG_MSG (CORETEXT, blob, "TTC index %u not supported", ttc_index);
    return nullptr; // CoreText does not support TTCs
  }

  if (unlikely (named_instance_index != 0))
  {
    auto ct_font_desc_array = CTFontManagerCreateFontDescriptorsFromData (CFDataCreate (kCFAllocatorDefault, (const UInt8 *) blob_data, blob_length));
    if (unlikely (!ct_font_desc_array))
      return nullptr;
    return create_cg_font (ct_font_desc_array, named_instance_index);
  }

  hb_blob_reference (blob);
  CGDataProviderRef provider = CGDataProviderCreateWithData (blob, blob_data, blob_length, &release_data);
  CGFontRef cg_font = nullptr;
  if (likely (provider))
  {
    cg_font = CGFontCreateWithDataProvider (provider);
    if (unlikely (!cg_font))
      DEBUG_MSG (CORETEXT, blob, "CGFontCreateWithDataProvider() failed");
    CGDataProviderRelease (provider);
  }
  return cg_font;
}

CGFontRef
create_cg_font (hb_face_t *face)
{
  CGFontRef cg_font = nullptr;
  if (face->destroy == _hb_cg_font_release)
    cg_font = CGFontRetain ((CGFontRef) face->user_data);
  else
  {
    hb_blob_t *blob = hb_face_reference_blob (face);
    cg_font = create_cg_font (blob, face->index);
    hb_blob_destroy (blob);
  }
  return cg_font;
}

CTFontRef
create_ct_font (CGFontRef cg_font, CGFloat font_size)
{
  CTFontRef ct_font = nullptr;

  /* CoreText does not enable trak table usage / tracking when creating a CTFont
   * using CTFontCreateWithGraphicsFont. The only way of enabling tracking seems
   * to be through the CTFontCreateUIFontForLanguage call. */
  CFStringRef cg_postscript_name = CGFontCopyPostScriptName (cg_font);
  if (CFStringHasPrefix (cg_postscript_name, CFSTR (".SFNSText")) ||
      CFStringHasPrefix (cg_postscript_name, CFSTR (".SFNSDisplay")))
  {
#if !(defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) && MAC_OS_X_VERSION_MIN_REQUIRED < 1080
# define kCTFontUIFontSystem kCTFontSystemFontType
# define kCTFontUIFontEmphasizedSystem kCTFontEmphasizedSystemFontType
#endif
    CTFontUIFontType font_type = kCTFontUIFontSystem;
    if (CFStringHasSuffix (cg_postscript_name, CFSTR ("-Bold")))
      font_type = kCTFontUIFontEmphasizedSystem;

    ct_font = CTFontCreateUIFontForLanguage (font_type, font_size, nullptr);
    CFStringRef ct_result_name = CTFontCopyPostScriptName(ct_font);
    if (CFStringCompare (ct_result_name, cg_postscript_name, 0) != kCFCompareEqualTo)
    {
      CFRelease(ct_font);
      ct_font = nullptr;
    }
    CFRelease (ct_result_name);
  }
  CFRelease (cg_postscript_name);

  if (!ct_font)
    ct_font = CTFontCreateWithGraphicsFont (cg_font, font_size, nullptr, nullptr);

  if (unlikely (!ct_font)) {
    DEBUG_MSG (CORETEXT, cg_font, "Font CTFontCreateWithGraphicsFont() failed");
    return nullptr;
  }

  /* crbug.com/576941 and crbug.com/625902 and the investigation in the latter
   * bug indicate that the cascade list reconfiguration occasionally causes
   * crashes in CoreText on OS X 10.9, thus let's skip this step on older
   * operating system versions. Except for the emoji font, where _not_
   * reconfiguring the cascade list causes CoreText crashes. For details, see
   * crbug.com/549610 */
  // 0x00070000 stands for "kCTVersionNumber10_10", see CoreText.h
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  if (&CTGetCoreTextVersion != nullptr && CTGetCoreTextVersion() < 0x00070000) {
#pragma GCC diagnostic pop
    CFStringRef fontName = CTFontCopyPostScriptName (ct_font);
    bool isEmojiFont = CFStringCompare (fontName, CFSTR("AppleColorEmoji"), 0) == kCFCompareEqualTo;
    CFRelease (fontName);
    if (!isEmojiFont)
      return ct_font;
  }

  CFURLRef original_url = nullptr;
#if !(defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) && MAC_OS_X_VERSION_MIN_REQUIRED < 1060
  ATSFontRef atsFont;
  FSRef fsref;
  OSStatus status;
  atsFont = CTFontGetPlatformFont (ct_font, NULL);
  status = ATSFontGetFileReference (atsFont, &fsref);
  if (status == noErr)
    original_url = CFURLCreateFromFSRef (NULL, &fsref);
#else
  original_url = (CFURLRef) CTFontCopyAttribute (ct_font, kCTFontURLAttribute);
#endif

  /* Create font copy with cascade list that has LastResort first; this speeds up CoreText
   * font fallback which we don't need anyway. */
  {
    CTFontDescriptorRef last_resort_font_desc = get_last_resort_font_desc ();
    CTFontRef new_ct_font = CTFontCreateCopyWithAttributes (ct_font, 0.0, nullptr, last_resort_font_desc);
    CFRelease (last_resort_font_desc);
    if (new_ct_font)
    {
      /* The CTFontCreateCopyWithAttributes call fails to stay on the same font
       * when reconfiguring the cascade list and may switch to a different font
       * when there are fonts that go by the same name, since the descriptor is
       * just name and size.
       *
       * Avoid reconfiguring the cascade lists if the new font is outside the
       * system locations that we cannot access from the sandboxed renderer
       * process in Blink. This can be detected by the new file URL location
       * that the newly found font points to. */
      CFURLRef new_url = nullptr;
#if !(defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE) && MAC_OS_X_VERSION_MIN_REQUIRED < 1060
      atsFont = CTFontGetPlatformFont (new_ct_font, NULL);
      status = ATSFontGetFileReference (atsFont, &fsref);
      if (status == noErr)
	new_url = CFURLCreateFromFSRef (NULL, &fsref);
#else
      new_url = (CFURLRef) CTFontCopyAttribute (new_ct_font, kCTFontURLAttribute);
#endif
      // Keep reconfigured font if URL cannot be retrieved (seems to be the case
      // on Mac OS 10.12 Sierra), speculative fix for crbug.com/625606
      if (!original_url || !new_url || CFEqual (original_url, new_url)) {
	CFRelease (ct_font);
	ct_font = new_ct_font;
      } else {
	CFRelease (new_ct_font);
	DEBUG_MSG (CORETEXT, ct_font, "Discarding reconfigured CTFont, location changed.");
      }
      if (new_url)
	CFRelease (new_url);
    }
    else
      DEBUG_MSG (CORETEXT, ct_font, "Font copy with empty cascade list failed");
  }

  if (original_url)
    CFRelease (original_url);
  return ct_font;
}

/**
 * hb_coretext_face_create:
 * @cg_font: The CGFontRef to work upon
 *
 * Creates an #hb_face_t face object from the specified
 * CGFontRef.
 *
 * Return value: (transfer full): The new face object
 *
 * Since: 0.9.10
 */
hb_face_t *
hb_coretext_face_create (CGFontRef cg_font)
{
  hb_face_t *face = hb_face_create_for_tables (_hb_cg_reference_table, CGFontRetain (cg_font), _hb_cg_font_release);
  hb_face_set_get_table_tags_func (face, _hb_cg_get_table_tags, cg_font, nullptr);
  return face;
}

/**
 * hb_coretext_face_create_from_file_or_fail:
 * @file_name: A font filename
 * @index: The index of the face within the file
 *
 * Creates an #hb_face_t face object from the specified
 * font file and face index.
 *
 * This is similar in functionality to hb_face_create_from_file_or_fail(),
 * but uses the CoreText library for loading the font file.
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * no face is found at the specified index or the file cannot be read.
 *
 * Since: 10.1.0
 */
hb_face_t *
hb_coretext_face_create_from_file_or_fail (const char   *file_name,
					   unsigned int  index)
{
  auto url = CFURLCreateFromFileSystemRepresentation (nullptr,
						      (const UInt8 *) file_name,
						      strlen (file_name),
						      false);
  if (unlikely (!url))
    return nullptr;

  auto ct_font_desc_array = CTFontManagerCreateFontDescriptorsFromURL (url);
  if (unlikely (!ct_font_desc_array))
  {
    CFRelease (url);
    return nullptr;
  }

  unsigned ttc_index = index & 0xFFFF;
  unsigned named_instance_index = index >> 16;

  if (ttc_index != 0)
  {
    DEBUG_MSG (CORETEXT, nullptr, "TTC index %u not supported", ttc_index);
    return nullptr; // CoreText does not support TTCs
  }

  auto cg_font = create_cg_font (ct_font_desc_array, named_instance_index);
  CFRelease (url);

  hb_face_t *face = hb_coretext_face_create (cg_font);
  CFRelease (cg_font);
  if (unlikely (hb_face_is_immutable (face)))
    return nullptr;

  hb_face_set_index (face, index);

  return face;
}

/**
 * hb_coretext_face_create_from_blob_or_fail:
 * @blob: A blob containing the font data
 * @index: The index of the face within the blob
 *
 * Creates an #hb_face_t face object from the specified
 * blob and face index.
 *
 * This is similar in functionality to hb_face_create_from_blob_or_fail(),
 * but uses the CoreText library for loading the font data.
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * no face is found at the specified index or the blob cannot be read.
 *
 * Since: 11.0.0
 */
hb_face_t *
hb_coretext_face_create_from_blob_or_fail (hb_blob_t    *blob,
					   unsigned int  index)
{
  auto cg_font = create_cg_font (blob, index);
  if (unlikely (!cg_font))
    return nullptr;

  hb_face_t *face = hb_coretext_face_create (cg_font);
  CFRelease (cg_font);
  if (unlikely (hb_face_is_immutable (face)))
    return nullptr;

  hb_face_set_index (face, index);

  return face;
}

/**
 * hb_coretext_face_get_cg_font:
 * @face: The #hb_face_t to work upon
 *
 * Fetches the CGFontRef associated with an #hb_face_t
 * face object
 *
 * Return value: the CGFontRef found
 *
 * Since: 0.9.10
 */
CGFontRef
hb_coretext_face_get_cg_font (hb_face_t *face)
{
  return (CGFontRef) (const void *) face->data.coretext;
}

/**
 * hb_coretext_font_create:
 * @ct_font: The CTFontRef to work upon
 *
 * Creates an #hb_font_t font object from the specified
 * CTFontRef.
 *
 * The created font uses the default font functions implemented
 * natively by HarfBuzz. If you want to use the CoreText font functions
 * instead (rarely needed), you can do so by calling
 * by hb_coretext_font_set_funcs().
 *
 * Return value: (transfer full): The new font object
 *
 * Since: 1.7.2
 **/
hb_font_t *
hb_coretext_font_create (CTFontRef ct_font)
{
  CGFontRef cg_font = CTFontCopyGraphicsFont (ct_font, nullptr);
  hb_face_t *face = hb_coretext_face_create (cg_font);
  CFRelease (cg_font);
  hb_font_t *font = hb_font_create (face);
  hb_face_destroy (face);

  if (unlikely (hb_object_is_immutable (font)))
    return font;

  hb_font_set_ptem (font, CTFontGetSize (ct_font));

  /* Copy font variations */
  CFDictionaryRef variations = CTFontCopyVariation (ct_font);
  if (variations)
  {
    hb_vector_t<hb_variation_t> vars;
    hb_vector_t<CFTypeRef> keys;
    hb_vector_t<CFTypeRef> values;

    CFIndex count = CFDictionaryGetCount (variations);
    if (unlikely (!vars.alloc_exact (count) || !keys.resize_exact (count) || !values.resize_exact (count)))
      goto done;

    // Fetch them one by one and collect in a vector of our own.
    CFDictionaryGetKeysAndValues (variations, keys.arrayZ, values.arrayZ);
    for (CFIndex i = 0; i < count; i++)
    {
      int tag;
      float value;
      CFNumberGetValue ((CFNumberRef) keys.arrayZ[i], kCFNumberIntType, &tag);
      CFNumberGetValue ((CFNumberRef) values.arrayZ[i], kCFNumberFloatType, &value);

      hb_variation_t var = {tag, value};
      vars.push (var);
    }
    hb_font_set_variations (font, vars.arrayZ, vars.length);

done:
    CFRelease (variations);
  }

  /* Let there be dragons here... */
  font->data.coretext.cmpexch (nullptr, (hb_coretext_font_data_t *) CFRetain (ct_font));

  // https://github.com/harfbuzz/harfbuzz/pull/4895#issuecomment-2408471254
  //hb_coretext_font_set_funcs (font);

  return font;
}

/**
 * hb_coretext_font_get_ct_font:
 * @font: #hb_font_t to work upon
 *
 * Fetches the CTFontRef associated with the specified
 * #hb_font_t font object.
 *
 * Return value: the CTFontRef found
 *
 * Since: 0.9.10
 */
CTFontRef
hb_coretext_font_get_ct_font (hb_font_t *font)
{
  return (CTFontRef) (const void *) font->data.coretext;
}


#endif
