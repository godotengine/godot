/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2012  Google, Inc.
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

#include "hb-face.hh"
#include "hb-blob.hh"
#include "hb-open-file.hh"
#include "hb-ot-face.hh"
#include "hb-ot-cmap-table.hh"

#ifdef HAVE_FREETYPE
#include "hb-ft.h"
#endif
#ifdef HAVE_CORETEXT
#include "hb-coretext.h"
#endif
#ifdef HAVE_DIRECTWRITE
#include "hb-directwrite.h"
#endif


/**
 * SECTION:hb-face
 * @title: hb-face
 * @short_description: Font face objects
 * @include: hb.h
 *
 * A font face is an object that represents a single face from within a
 * font family.
 *
 * More precisely, a font face represents a single face in a binary font file.
 * Font faces are typically built from a binary blob and a face index.
 * Font faces are used to create fonts.
 *
 * A font face can be created from a binary blob using hb_face_create().
 * The face index is used to select a face from a binary blob that contains
 * multiple faces.  For example, a binary blob that contains both a regular
 * and a bold face can be used to create two font faces, one for each face
 * index.
 **/


/**
 * hb_face_count:
 * @blob: a blob.
 *
 * Fetches the number of faces in a blob.
 *
 * Return value: Number of faces in @blob
 *
 * Since: 1.7.7
 **/
unsigned int
hb_face_count (hb_blob_t *blob)
{
  if (unlikely (!blob))
    return 0;

  hb_sanitize_context_t c (blob);

  auto *ot = blob->as<OT::OpenTypeFontFile> ();
  if (unlikely (!ot->sanitize (&c)))
    return 0;

  return ot->get_face_count ();
}

/*
 * hb_face_t
 */

DEFINE_NULL_INSTANCE (hb_face_t) =
{
  HB_OBJECT_HEADER_STATIC,

  0,    /* index */
  1000, /* upem */
  0,    /* num_glyphs */

  /* Zero for the rest is fine. */
};


/**
 * hb_face_create_for_tables:
 * @reference_table_func: (closure user_data) (destroy destroy) (scope notified): Table-referencing function
 * @user_data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 *
 * Variant of hb_face_create(), built for those cases where it is more
 * convenient to provide data for individual tables instead of the whole font
 * data. With the caveat that hb_face_get_table_tags() would not work
 * with faces created this way. You can address that by calling the
 * hb_face_set_get_table_tags_func() function and setting the appropriate callback.
 *
 * Creates a new face object from the specified @user_data and @reference_table_func,
 * with the @destroy callback.
 *
 * Return value: (transfer full): The new face object
 *
 * Since: 0.9.2
 **/
hb_face_t *
hb_face_create_for_tables (hb_reference_table_func_t  reference_table_func,
			   void                      *user_data,
			   hb_destroy_func_t          destroy)
{
  hb_face_t *face;

  if (!reference_table_func || !(face = hb_object_create<hb_face_t> ())) {
    if (destroy)
      destroy (user_data);
    return hb_face_get_empty ();
  }

  face->reference_table_func = reference_table_func;
  face->user_data = user_data;
  face->destroy = destroy;

  face->num_glyphs = -1;

  face->data.init0 (face);
  face->table.init0 (face);

  return face;
}


typedef struct hb_face_for_data_closure_t {
  hb_blob_t *blob;
  uint16_t  index;
} hb_face_for_data_closure_t;

static hb_face_for_data_closure_t *
_hb_face_for_data_closure_create (hb_blob_t *blob, unsigned int index)
{
  hb_face_for_data_closure_t *closure;

  closure = (hb_face_for_data_closure_t *) hb_calloc (1, sizeof (hb_face_for_data_closure_t));
  if (unlikely (!closure))
    return nullptr;

  closure->blob = blob;
  closure->index = (uint16_t) (index & 0xFFFFu);

  return closure;
}

static void
_hb_face_for_data_closure_destroy (void *data)
{
  hb_face_for_data_closure_t *closure = (hb_face_for_data_closure_t *) data;

  hb_blob_destroy (closure->blob);
  hb_free (closure);
}

static hb_blob_t *
_hb_face_for_data_reference_table (hb_face_t *face HB_UNUSED, hb_tag_t tag, void *user_data)
{
  hb_face_for_data_closure_t *data = (hb_face_for_data_closure_t *) user_data;

  if (tag == HB_TAG_NONE)
    return hb_blob_reference (data->blob);

  const OT::OpenTypeFontFile &ot_file = *data->blob->as<OT::OpenTypeFontFile> ();
  unsigned int base_offset;
  const OT::OpenTypeFontFace &ot_face = ot_file.get_face (data->index, &base_offset);

  const OT::OpenTypeTable &table = ot_face.get_table_by_tag (tag);

  hb_blob_t *blob = hb_blob_create_sub_blob (data->blob, base_offset + table.offset, table.length);

  return blob;
}

static unsigned
_hb_face_for_data_get_table_tags (const hb_face_t *face HB_UNUSED,
				  unsigned int start_offset,
				  unsigned int *table_count,
				  hb_tag_t *table_tags,
				  void *user_data)
{
  hb_face_for_data_closure_t *data = (hb_face_for_data_closure_t *) user_data;

  const OT::OpenTypeFontFile &ot_file = *data->blob->as<OT::OpenTypeFontFile> ();
  const OT::OpenTypeFontFace &ot_face = ot_file.get_face (data->index);

  return ot_face.get_table_tags (start_offset, table_count, table_tags);
}


/**
 * hb_face_create:
 * @blob: #hb_blob_t to work upon
 * @index: The index of the face within @blob
 *
 * Constructs a new face object from the specified blob and
 * a face index into that blob.
 *
 * The face index is used for blobs of file formats such as TTC and
 * DFont that can contain more than one face.  Face indices within
 * such collections are zero-based.
 *
 * <note>Note: If the blob font format is not a collection, @index
 * is ignored.  Otherwise, only the lower 16-bits of @index are used.
 * The unmodified @index can be accessed via hb_face_get_index().</note>
 *
 * <note>Note: The high 16-bits of @index, if non-zero, are used by
 * hb_font_create() to load named-instances in variable fonts.  See
 * hb_font_create() for details.</note>
 *
 * Return value: (transfer full): The new face object
 *
 * Since: 0.9.2
 **/
hb_face_t *
hb_face_create (hb_blob_t    *blob,
		unsigned int  index)
{
  hb_face_t *face;

  if (unlikely (!blob))
    blob = hb_blob_get_empty ();

  blob = hb_sanitize_context_t ().sanitize_blob<OT::OpenTypeFontFile> (hb_blob_reference (blob));

  hb_face_for_data_closure_t *closure = _hb_face_for_data_closure_create (blob, index);

  if (unlikely (!closure))
  {
    hb_blob_destroy (blob);
    return hb_face_get_empty ();
  }

  face = hb_face_create_for_tables (_hb_face_for_data_reference_table,
				    closure,
				    _hb_face_for_data_closure_destroy);
  hb_face_set_get_table_tags_func (face,
				   _hb_face_for_data_get_table_tags,
				   closure,
				   nullptr);

  face->index = index;

  return face;
}

/**
 * hb_face_create_or_fail:
 * @blob: #hb_blob_t to work upon
 * @index: The index of the face within @blob
 *
 * Like hb_face_create(), but returns `NULL` if the blob data
 * contains no usable font face at the specified index.
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * no face is found at the specified index.
 *
 * Since: 10.1.0
 **/
hb_face_t *
hb_face_create_or_fail (hb_blob_t    *blob,
			unsigned int  index)
{
  unsigned num_faces = hb_face_count (blob);
  if (index >= num_faces)
    return nullptr;

  hb_face_t *face = hb_face_create (blob, index);
  if (hb_object_is_immutable (face))
    return nullptr;

  return face;
}

#ifndef HB_NO_OPEN
/**
 * hb_face_create_from_file_or_fail:
 * @file_name: A font filename
 * @index: The index of the face within the file
 *
 * A thin wrapper around hb_blob_create_from_file_or_fail()
 * followed by hb_face_create_or_fail().
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * no face is found at the specified index or the file cannot be read.
 *
 * Since: 10.1.0
 **/
HB_EXTERN hb_face_t *
hb_face_create_from_file_or_fail (const char   *file_name,
				  unsigned int  index)
{
  hb_blob_t *blob = hb_blob_create_from_file_or_fail (file_name);
  if (unlikely (!blob))
    return nullptr;

  hb_face_t *face = hb_face_create_or_fail (blob, index);
  hb_blob_destroy (blob);

  return face;
}

static const struct supported_face_loaders_t {
	char name[16];
	hb_face_t * (*from_file) (const char *font_file, unsigned face_index);
	hb_face_t * (*from_blob) (hb_blob_t *blob, unsigned face_index);
} supported_face_loaders[] =
{
  {"ot",
#ifndef HB_NO_OPEN
   hb_face_create_from_file_or_fail,
#else
   nullptr,
#endif
   hb_face_create_or_fail
  },
#ifdef HAVE_FREETYPE
  {"ft",
   hb_ft_face_create_from_file_or_fail,
   hb_ft_face_create_from_blob_or_fail
  },
#endif
#ifdef HAVE_CORETEXT
  {"coretext",
   hb_coretext_face_create_from_file_or_fail,
   hb_coretext_face_create_from_blob_or_fail
  },
#endif
#ifdef HAVE_DIRECTWRITE
  {"directwrite",
   hb_directwrite_face_create_from_file_or_fail,
   hb_directwrite_face_create_from_blob_or_fail
  },
#endif
};

static const char *get_default_loader_name ()
{
  static hb_atomic_t<const char *> static_loader_name;
  const char *loader_name = static_loader_name.get_acquire ();
  if (!loader_name)
  {
    loader_name = getenv ("HB_FACE_LOADER");
    if (!loader_name)
      loader_name = "";
    if (!static_loader_name.cmpexch (nullptr, loader_name))
      loader_name = static_loader_name.get_acquire ();
  }
  return loader_name;
}

/**
 * hb_face_create_from_file_or_fail_using:
 * @file_name: A font filename
 * @index: The index of the face within the file
 * @loader_name: (nullable): The name of the loader to use, or `NULL`
 *
 * A thin wrapper around the face loader functions registered with HarfBuzz.
 * If @loader_name is `NULL` or the empty string, the first available loader
 * is used.
 *
 * For example, the FreeType ("ft") loader might be able to load
 * WOFF and WOFF2 files if FreeType is built with those features,
 * whereas the OpenType ("ot") loader will not.
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * the file cannot be read or the loader fails to load the face.
 *
 * Since: 11.0.0
 **/
hb_face_t *
hb_face_create_from_file_or_fail_using (const char   *file_name,
					unsigned int  index,
					const char   *loader_name)
{
  // Duplicated in hb_face_create_or_fail_using
  bool retry = false;
  if (!loader_name || !*loader_name)
  {
    loader_name = get_default_loader_name ();
    retry = true;
  }
  if (loader_name && !*loader_name) loader_name = nullptr;

retry:
  for (unsigned i = 0; i < ARRAY_LENGTH (supported_face_loaders); i++)
  {
    if (!loader_name || (supported_face_loaders[i].from_file && !strcmp (supported_face_loaders[i].name, loader_name)))
      return supported_face_loaders[i].from_file (file_name, index);
  }

  if (retry)
  {
    retry = false;
    loader_name = nullptr;
    goto retry;
  }

  return nullptr;
}

/**
 * hb_face_create_or_fail_using:
 * @blob: #hb_blob_t to work upon
 * @index: The index of the face within @blob
 * @loader_name: (nullable): The name of the loader to use, or `NULL`
 *
 * A thin wrapper around the face loader functions registered with HarfBuzz.
 * If @loader_name is `NULL` or the empty string, the first available loader
 * is used.
 *
 * For example, the FreeType ("ft") loader might be able to load
 * WOFF and WOFF2 files if FreeType is built with those features,
 * whereas the OpenType ("ot") loader will not.
 *
 * Return value: (transfer full): The new face object, or `NULL` if
 * the loader fails to load the face.
 *
 * Since: 11.0.0
 **/
hb_face_t *
hb_face_create_or_fail_using (hb_blob_t    *blob,
			      unsigned int  index,
			      const char   *loader_name)
{
  // Duplicated in hb_face_create_from_file_or_fail_using
  bool retry = false;
  if (!loader_name || !*loader_name)
  {
    loader_name = get_default_loader_name ();
    retry = true;
  }
  if (loader_name && !*loader_name) loader_name = nullptr;

retry:
  for (unsigned i = 0; i < ARRAY_LENGTH (supported_face_loaders); i++)
  {
    if (!loader_name || (supported_face_loaders[i].from_blob && !strcmp (supported_face_loaders[i].name, loader_name)))
      return supported_face_loaders[i].from_blob (blob, index);
  }

  if (retry)
  {
    retry = false;
    loader_name = nullptr;
    goto retry;
  }

  return nullptr;
}

static inline void free_static_face_loader_list ();

static const char * const nil_face_loader_list[] = {nullptr};

static struct hb_face_loader_list_lazy_loader_t : hb_lazy_loader_t<const char *,
								  hb_face_loader_list_lazy_loader_t>
{
  static const char ** create ()
  {
    const char **face_loader_list = (const char **) hb_calloc (1 + ARRAY_LENGTH (supported_face_loaders), sizeof (const char *));
    if (unlikely (!face_loader_list))
      return nullptr;

    unsigned i;
    for (i = 0; i < ARRAY_LENGTH (supported_face_loaders); i++)
      face_loader_list[i] = supported_face_loaders[i].name;
    face_loader_list[i] = nullptr;

    hb_atexit (free_static_face_loader_list);

    return face_loader_list;
  }
  static void destroy (const char **l)
  { hb_free (l); }
  static const char * const * get_null ()
  { return nil_face_loader_list; }
} static_face_loader_list;

static inline
void free_static_face_loader_list ()
{
  static_face_loader_list.free_instance ();
}

/**
 * hb_face_list_loaders:
 *
 * Retrieves the list of face loaders supported by HarfBuzz.
 *
 * Return value: (transfer none) (array zero-terminated=1): a
 *    `NULL`-terminated array of supported face loaders
 *    constant strings. The returned array is owned by HarfBuzz
 *    and should not be modified or freed.
 *
 * Since: 11.0.0
 **/
const char **
hb_face_list_loaders ()
{
  return static_face_loader_list.get_unconst ();
}
#endif


/**
 * hb_face_get_empty:
 *
 * Fetches the singleton empty face object.
 *
 * Return value: (transfer full): The empty face object
 *
 * Since: 0.9.2
 **/
hb_face_t *
hb_face_get_empty ()
{
  return const_cast<hb_face_t *> (&Null (hb_face_t));
}


/**
 * hb_face_reference: (skip)
 * @face: A face object
 *
 * Increases the reference count on a face object.
 *
 * Return value: The @face object
 *
 * Since: 0.9.2
 **/
hb_face_t *
hb_face_reference (hb_face_t *face)
{
  return hb_object_reference (face);
}

/**
 * hb_face_destroy: (skip)
 * @face: A face object
 *
 * Decreases the reference count on a face object. When the
 * reference count reaches zero, the face is destroyed,
 * freeing all memory.
 *
 * Since: 0.9.2
 **/
void
hb_face_destroy (hb_face_t *face)
{
  if (!hb_object_destroy (face)) return;

#ifndef HB_NO_SHAPER
  for (hb_face_t::plan_node_t *node = face->shape_plans; node; )
  {
    hb_face_t::plan_node_t *next = node->next;
    hb_shape_plan_destroy (node->shape_plan);
    hb_free (node);
    node = next;
  }
#endif

  face->data.fini ();
  face->table.fini ();

  if (face->get_table_tags_destroy)
    face->get_table_tags_destroy (face->get_table_tags_user_data);

  if (face->destroy)
    face->destroy (face->user_data);

  hb_free (face);
}

/**
 * hb_face_set_user_data: (skip)
 * @face: A face object
 * @key: The user-data key to set
 * @data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 * @replace: Whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the given face object.
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_face_set_user_data (hb_face_t          *face,
		       hb_user_data_key_t *key,
		       void *              data,
		       hb_destroy_func_t   destroy,
		       hb_bool_t           replace)
{
  return hb_object_set_user_data (face, key, data, destroy, replace);
}

/**
 * hb_face_get_user_data: (skip)
 * @face: A face object
 * @key: The user-data key to query
 *
 * Fetches the user data associated with the specified key,
 * attached to the specified face object.
 *
 * Return value: (transfer none): A pointer to the user data
 *
 * Since: 0.9.2
 **/
void *
hb_face_get_user_data (const hb_face_t    *face,
		       hb_user_data_key_t *key)
{
  return hb_object_get_user_data (face, key);
}

/**
 * hb_face_make_immutable:
 * @face: A face object
 *
 * Makes the given face object immutable.
 *
 * Since: 0.9.2
 **/
void
hb_face_make_immutable (hb_face_t *face)
{
  if (hb_object_is_immutable (face))
    return;

  hb_object_make_immutable (face);
}

/**
 * hb_face_is_immutable:
 * @face: A face object
 *
 * Tests whether the given face object is immutable.
 *
 * Return value: `true` is @face is immutable, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_face_is_immutable (hb_face_t *face)
{
  return hb_object_is_immutable (face);
}


/**
 * hb_face_reference_table:
 * @face: A face object
 * @tag: The #hb_tag_t of the table to query
 *
 * Fetches a reference to the specified table within
 * the specified face. Returns an empty blob if referencing table data is not
 * possible.
 *
 * Return value: (transfer full): A pointer to the @tag table within @face
 *
 * Since: 0.9.2
 **/
hb_blob_t *
hb_face_reference_table (const hb_face_t *face,
			 hb_tag_t tag)
{
  if (unlikely (tag == HB_TAG_NONE))
    return hb_blob_get_empty ();

  return face->reference_table (tag);
}

/**
 * hb_face_reference_blob:
 * @face: A face object
 *
 * Fetches a pointer to the binary blob that contains the specified face.
 * If referencing the face data is not possible, this function creates a blob
 * out of individual table blobs if hb_face_get_table_tags() works with this
 * face, otherwise it returns an empty blob.
 *
 * Return value: (transfer full): A pointer to the blob for @face
 *
 * Since: 0.9.2
 **/
hb_blob_t *
hb_face_reference_blob (hb_face_t *face)
{
  hb_blob_t *blob = face->reference_table (HB_TAG_NONE);

  if (blob == hb_blob_get_empty ())
  {
    // If referencing the face blob is not possible (e.g. not implemented by the
    // font functions), use face builder to create a blob out of individual
    // table blobs.
    unsigned total_count = hb_face_get_table_tags (face, 0, nullptr, nullptr);
    if (total_count)
    {
      hb_tag_t tags[64];
      unsigned count = ARRAY_LENGTH (tags);
      hb_face_t* builder = hb_face_builder_create ();

      for (unsigned offset = 0; offset < total_count; offset += count)
      {
        hb_face_get_table_tags (face, offset, &count, tags);
	if (unlikely (!count))
	  break; // Allocation error
        for (unsigned i = 0; i < count; i++)
        {
	  if (unlikely (!tags[i]))
	    continue;
	  hb_blob_t *table = hb_face_reference_table (face, tags[i]);
	  hb_face_builder_add_table (builder, tags[i], table);
	  hb_blob_destroy (table);
        }
      }

      blob = hb_face_reference_blob (builder);
      hb_face_destroy (builder);
    }
  }

  return blob;
}

/**
 * hb_face_set_index:
 * @face: A face object
 * @index: The index to assign
 *
 * Assigns the specified face-index to @face. Fails if the
 * face is immutable.
 *
 * <note>Note: changing the index has no effect on the face itself
 * This only changes the value returned by hb_face_get_index().</note>
 *
 * Since: 0.9.2
 **/
void
hb_face_set_index (hb_face_t    *face,
		   unsigned int  index)
{
  if (hb_object_is_immutable (face))
    return;

  face->index = index;
}

/**
 * hb_face_get_index:
 * @face: A face object
 *
 * Fetches the face-index corresponding to the given face.
 *
 * <note>Note: face indices within a collection are zero-based.</note>
 *
 * Return value: The index of @face.
 *
 * Since: 0.9.2
 **/
unsigned int
hb_face_get_index (const hb_face_t *face)
{
  return face->index;
}

/**
 * hb_face_set_upem:
 * @face: A face object
 * @upem: The units-per-em value to assign
 *
 * Sets the units-per-em (upem) for a face object to the specified value.
 *
 * This API is used in rare circumstances.
 *
 * Since: 0.9.2
 **/
void
hb_face_set_upem (hb_face_t    *face,
		  unsigned int  upem)
{
  if (hb_object_is_immutable (face))
    return;

  face->upem = upem;
}

/**
 * hb_face_get_upem:
 * @face: A face object
 *
 * Fetches the units-per-em (UPEM) value of the specified face object.
 *
 * Typical UPEM values for fonts are 1000, or 2048, but any value
 * in between 16 and 16,384 is allowed for OpenType fonts.
 *
 * Return value: The upem value of @face
 *
 * Since: 0.9.2
 **/
unsigned int
hb_face_get_upem (const hb_face_t *face)
{
  return face->get_upem ();
}

/**
 * hb_face_set_glyph_count:
 * @face: A face object
 * @glyph_count: The glyph-count value to assign
 *
 * Sets the glyph count for a face object to the specified value.
 *
 * This API is used in rare circumstances.
 *
 * Since: 0.9.7
 **/
void
hb_face_set_glyph_count (hb_face_t    *face,
			 unsigned int  glyph_count)
{
  if (hb_object_is_immutable (face))
    return;

  face->num_glyphs = glyph_count;
}

/**
 * hb_face_get_glyph_count:
 * @face: A face object
 *
 * Fetches the glyph-count value of the specified face object.
 *
 * Return value: The glyph-count value of @face
 *
 * Since: 0.9.7
 **/
unsigned int
hb_face_get_glyph_count (const hb_face_t *face)
{
  return face->get_num_glyphs ();
}

/**
 * hb_face_set_get_table_tags_func:
 * @face: A face object
 * @func: (closure user_data) (destroy destroy) (scope notified): The table-tag-fetching function
 * @user_data: A pointer to the user data, to be destroyed by @destroy when not needed anymore
 * @destroy: (nullable): A callback to call when @func is not needed anymore
 *
 * Sets the table-tag-fetching function for the specified face object.
 *
 * Since: 10.0.0
 */
HB_EXTERN void
hb_face_set_get_table_tags_func (hb_face_t *face,
				 hb_get_table_tags_func_t func,
				 void                    *user_data,
				 hb_destroy_func_t        destroy)
{
  if (hb_object_is_immutable (face))
  {
    if (destroy)
      destroy (user_data);
    return;
  }

  if (face->get_table_tags_destroy)
    face->get_table_tags_destroy (face->get_table_tags_user_data);

  face->get_table_tags_func = func;
  face->get_table_tags_user_data = user_data;
  face->get_table_tags_destroy = destroy;
}

/**
 * hb_face_get_table_tags:
 * @face: A face object
 * @start_offset: The index of first table tag to retrieve
 * @table_count: (inout): Input = the maximum number of table tags to return;
 *                Output = the actual number of table tags returned (may be zero)
 * @table_tags: (out) (array length=table_count): The array of table tags found
 *
 * Fetches a list of all table tags for a face, if possible. The list returned will
 * begin at the offset provided
 *
 * Return value: Total number of tables, or zero if it is not possible to list
 *
 * Since: 1.6.0
 **/
unsigned int
hb_face_get_table_tags (const hb_face_t *face,
			unsigned int  start_offset,
			unsigned int *table_count, /* IN/OUT */
			hb_tag_t     *table_tags /* OUT */)
{
  if (!face->get_table_tags_func)
  {
    if (table_count)
      *table_count = 0;
    return 0;
  }

  return face->get_table_tags_func (face, start_offset, table_count, table_tags, face->get_table_tags_user_data);
}


/*
 * Character set.
 */


#ifndef HB_NO_FACE_COLLECT_UNICODES
/**
 * hb_face_collect_unicodes:
 * @face: A face object
 * @out: (out): The set to add Unicode characters to
 *
 * Collects all of the Unicode characters covered by @face and adds
 * them to the #hb_set_t set @out.
 *
 * Since: 1.9.0
 */
void
hb_face_collect_unicodes (hb_face_t *face,
			  hb_set_t  *out)
{
  face->table.cmap->collect_unicodes (out, face->get_num_glyphs ());
}
/**
 * hb_face_collect_nominal_glyph_mapping:
 * @face: A face object
 * @mapping: (out): The map to add Unicode-to-glyph mapping to
 * @unicodes: (nullable) (out): The set to add Unicode characters to, or `NULL`
 *
 * Collects the mapping from Unicode characters to nominal glyphs of the @face,
 * and optionally all of the Unicode characters covered by @face.
 *
 * Since: 7.0.0
 */
void
hb_face_collect_nominal_glyph_mapping (hb_face_t *face,
				       hb_map_t  *mapping,
				       hb_set_t  *unicodes)
{
  hb_set_t stack_unicodes;
  if (!unicodes)
    unicodes = &stack_unicodes;
  face->table.cmap->collect_mapping (unicodes, mapping, face->get_num_glyphs ());
}
/**
 * hb_face_collect_variation_selectors:
 * @face: A face object
 * @out: (out): The set to add Variation Selector characters to
 *
 * Collects all Unicode "Variation Selector" characters covered by @face and adds
 * them to the #hb_set_t set @out.
 *
 * Since: 1.9.0
 */
void
hb_face_collect_variation_selectors (hb_face_t *face,
				     hb_set_t  *out)
{
  face->table.cmap->collect_variation_selectors (out);
}
/**
 * hb_face_collect_variation_unicodes:
 * @face: A face object
 * @variation_selector: The Variation Selector to query
 * @out: (out): The set to add Unicode characters to
 *
 * Collects all Unicode characters for @variation_selector covered by @face and adds
 * them to the #hb_set_t set @out.
 *
 * Since: 1.9.0
 */
void
hb_face_collect_variation_unicodes (hb_face_t *face,
				    hb_codepoint_t variation_selector,
				    hb_set_t  *out)
{
  face->table.cmap->collect_variation_unicodes (variation_selector, out);
}
#endif
