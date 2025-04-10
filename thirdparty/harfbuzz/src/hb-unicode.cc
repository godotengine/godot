/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2011  Codethink Limited
 * Copyright © 2010,2011,2012  Google, Inc.
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
 * Codethink Author(s): Ryan Lortie
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#include "hb-unicode.hh"


/**
 * SECTION: hb-unicode
 * @title: hb-unicode
 * @short_description: Unicode character property access
 * @include: hb.h
 *
 * Unicode functions are used to access Unicode character properties.
 * With these functions, client programs can query various properties from
 * the Unicode Character Database for any code point, such as General
 * Category (gc), Script (sc), Canonical Combining Class (ccc), etc.
 *
 * Client programs can optionally pass in their own Unicode functions
 * that implement the same queries. The set of functions available is
 * defined by the virtual methods in #hb_unicode_funcs_t.
 *
 * HarfBuzz provides built-in default functions for each method in
 * #hb_unicode_funcs_t.
 **/


/*
 * hb_unicode_funcs_t
 */

static hb_unicode_combining_class_t
hb_unicode_combining_class_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
				hb_codepoint_t      unicode   HB_UNUSED,
				void               *user_data HB_UNUSED)
{
  return HB_UNICODE_COMBINING_CLASS_NOT_REORDERED;
}

#ifndef HB_DISABLE_DEPRECATED
static unsigned int
hb_unicode_eastasian_width_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
				hb_codepoint_t      unicode   HB_UNUSED,
				void               *user_data HB_UNUSED)
{
  return 1;
}
#endif

static hb_unicode_general_category_t
hb_unicode_general_category_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
				 hb_codepoint_t      unicode   HB_UNUSED,
				 void               *user_data HB_UNUSED)
{
  return HB_UNICODE_GENERAL_CATEGORY_OTHER_LETTER;
}

static hb_codepoint_t
hb_unicode_mirroring_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
			  hb_codepoint_t      unicode,
			  void               *user_data HB_UNUSED)
{
  return unicode;
}

static hb_script_t
hb_unicode_script_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
		       hb_codepoint_t      unicode   HB_UNUSED,
		       void               *user_data HB_UNUSED)
{
  return HB_SCRIPT_UNKNOWN;
}

static hb_bool_t
hb_unicode_compose_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
			hb_codepoint_t      a         HB_UNUSED,
			hb_codepoint_t      b         HB_UNUSED,
			hb_codepoint_t     *ab        HB_UNUSED,
			void               *user_data HB_UNUSED)
{
  return false;
}

static hb_bool_t
hb_unicode_decompose_nil (hb_unicode_funcs_t *ufuncs    HB_UNUSED,
			  hb_codepoint_t      ab        HB_UNUSED,
			  hb_codepoint_t     *a         HB_UNUSED,
			  hb_codepoint_t     *b         HB_UNUSED,
			  void               *user_data HB_UNUSED)
{
  return false;
}


#ifndef HB_DISABLE_DEPRECATED
static unsigned int
hb_unicode_decompose_compatibility_nil (hb_unicode_funcs_t *ufuncs     HB_UNUSED,
					hb_codepoint_t      u          HB_UNUSED,
					hb_codepoint_t     *decomposed HB_UNUSED,
					void               *user_data  HB_UNUSED)
{
  return 0;
}
#endif

#if !defined(HB_NO_UNICODE_FUNCS) && defined(HAVE_GLIB)
#include "hb-glib.h"
#endif
#if !defined(HB_NO_UNICODE_FUNCS) && defined(HAVE_ICU) && defined(HAVE_ICU_BUILTIN)
#include "hb-icu.h"
#endif

/**
 * hb_unicode_funcs_get_default:
 *
 * Fetches a pointer to the default Unicode-functions structure that is used
 * when no functions are explicitly set on #hb_buffer_t.
 *
 * Return value: (transfer none): a pointer to the #hb_unicode_funcs_t Unicode-functions structure
 *
 * Since: 0.9.2
 **/
hb_unicode_funcs_t *
hb_unicode_funcs_get_default ()
{
#if !defined(HB_NO_UNICODE_FUNCS) && !defined(HB_NO_UCD)
  return hb_ucd_get_unicode_funcs ();
#elif !defined(HB_NO_UNICODE_FUNCS) && defined(HAVE_GLIB)
  return hb_glib_get_unicode_funcs ();
#elif !defined(HB_NO_UNICODE_FUNCS) && defined(HAVE_ICU) && defined(HAVE_ICU_BUILTIN)
  return hb_icu_get_unicode_funcs ();
#else
#define HB_UNICODE_FUNCS_NIL 1
  return hb_unicode_funcs_get_empty ();
#endif
}

#if !defined(HB_NO_UNICODE_FUNCS) && defined(HB_UNICODE_FUNCS_NIL)
#error "Could not find any Unicode functions implementation, you have to provide your own"
#error "Consider building hb-ucd.cc.  If you absolutely want to build without any, define HB_NO_UNICODE_FUNCS."
#endif

/**
 * hb_unicode_funcs_create:
 * @parent: (nullable): Parent Unicode-functions structure
 *
 * Creates a new #hb_unicode_funcs_t structure of Unicode functions.
 *
 * Return value: (transfer full): The Unicode-functions structure
 *
 * Since: 0.9.2
 **/
hb_unicode_funcs_t *
hb_unicode_funcs_create (hb_unicode_funcs_t *parent)
{
  hb_unicode_funcs_t *ufuncs;

  if (!(ufuncs = hb_object_create<hb_unicode_funcs_t> ()))
    return hb_unicode_funcs_get_empty ();

  if (!parent)
    parent = hb_unicode_funcs_get_empty ();

  hb_unicode_funcs_make_immutable (parent);
  ufuncs->parent = hb_unicode_funcs_reference (parent);

  ufuncs->func = parent->func;

  /* We can safely copy user_data from parent since we hold a reference
   * onto it and it's immutable.  We should not copy the destroy notifiers
   * though. */
  ufuncs->user_data = parent->user_data;

  return ufuncs;
}


DEFINE_NULL_INSTANCE (hb_unicode_funcs_t) =
{
  HB_OBJECT_HEADER_STATIC,

  nullptr, /* parent */
  {
#define HB_UNICODE_FUNC_IMPLEMENT(name) hb_unicode_##name##_nil,
    HB_UNICODE_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_UNICODE_FUNC_IMPLEMENT
  }
};

/**
 * hb_unicode_funcs_get_empty:
 *
 * Fetches the singleton empty Unicode-functions structure.
 *
 * Return value: (transfer full): The empty Unicode-functions structure
 *
 * Since: 0.9.2
 **/
hb_unicode_funcs_t *
hb_unicode_funcs_get_empty ()
{
  return const_cast<hb_unicode_funcs_t *> (&Null (hb_unicode_funcs_t));
}

/**
 * hb_unicode_funcs_reference: (skip)
 * @ufuncs: The Unicode-functions structure
 *
 * Increases the reference count on a Unicode-functions structure.
 *
 * Return value: (transfer full): The Unicode-functions structure
 *
 * Since: 0.9.2
 **/
hb_unicode_funcs_t *
hb_unicode_funcs_reference (hb_unicode_funcs_t *ufuncs)
{
  return hb_object_reference (ufuncs);
}

/**
 * hb_unicode_funcs_destroy: (skip)
 * @ufuncs: The Unicode-functions structure
 *
 * Decreases the reference count on a Unicode-functions structure. When
 * the reference count reaches zero, the Unicode-functions structure is
 * destroyed, freeing all memory.
 *
 * Since: 0.9.2
 **/
void
hb_unicode_funcs_destroy (hb_unicode_funcs_t *ufuncs)
{
  if (!hb_object_destroy (ufuncs)) return;

#define HB_UNICODE_FUNC_IMPLEMENT(name) \
  if (ufuncs->destroy.name) ufuncs->destroy.name (ufuncs->user_data.name);
    HB_UNICODE_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_UNICODE_FUNC_IMPLEMENT

  hb_unicode_funcs_destroy (ufuncs->parent);

  hb_free (ufuncs);
}

/**
 * hb_unicode_funcs_set_user_data: (skip)
 * @ufuncs: The Unicode-functions structure
 * @key: The user-data key
 * @data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 * @replace: Whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the specified Unicode-functions structure. 
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_unicode_funcs_set_user_data (hb_unicode_funcs_t *ufuncs,
				hb_user_data_key_t *key,
				void *              data,
				hb_destroy_func_t   destroy,
				hb_bool_t           replace)
{
  return hb_object_set_user_data (ufuncs, key, data, destroy, replace);
}

/**
 * hb_unicode_funcs_get_user_data: (skip)
 * @ufuncs: The Unicode-functions structure
 * @key: The user-data key to query
 *
 * Fetches the user-data associated with the specified key,
 * attached to the specified Unicode-functions structure.
 *
 * Return value: (transfer none): A pointer to the user data
 *
 * Since: 0.9.2
 **/
void *
hb_unicode_funcs_get_user_data (const hb_unicode_funcs_t *ufuncs,
				hb_user_data_key_t       *key)
{
  return hb_object_get_user_data (ufuncs, key);
}


/**
 * hb_unicode_funcs_make_immutable:
 * @ufuncs: The Unicode-functions structure
 *
 * Makes the specified Unicode-functions structure
 * immutable.
 *
 * Since: 0.9.2
 **/
void
hb_unicode_funcs_make_immutable (hb_unicode_funcs_t *ufuncs)
{
  if (hb_object_is_immutable (ufuncs))
    return;

  hb_object_make_immutable (ufuncs);
}

/**
 * hb_unicode_funcs_is_immutable:
 * @ufuncs: The Unicode-functions structure
 *
 * Tests whether the specified Unicode-functions structure
 * is immutable.
 *
 * Return value: `true` if @ufuncs is immutable, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_unicode_funcs_is_immutable (hb_unicode_funcs_t *ufuncs)
{
  return hb_object_is_immutable (ufuncs);
}

/**
 * hb_unicode_funcs_get_parent:
 * @ufuncs: The Unicode-functions structure
 *
 * Fetches the parent of the Unicode-functions structure
 * @ufuncs.
 *
 * Return value: The parent Unicode-functions structure
 *
 * Since: 0.9.2
 **/
hb_unicode_funcs_t *
hb_unicode_funcs_get_parent (hb_unicode_funcs_t *ufuncs)
{
  return ufuncs->parent ? ufuncs->parent : hb_unicode_funcs_get_empty ();
}


#define HB_UNICODE_FUNC_IMPLEMENT(name)						\
										\
void										\
hb_unicode_funcs_set_##name##_func (hb_unicode_funcs_t		   *ufuncs,	\
				    hb_unicode_##name##_func_t	    func,	\
				    void			   *user_data,	\
				    hb_destroy_func_t		    destroy)	\
{										\
  if (hb_object_is_immutable (ufuncs))						\
    goto fail;									\
										\
  if (!func)									\
  {										\
    if (destroy)								\
      destroy (user_data);							\
    destroy = nullptr;								\
    user_data = ufuncs->parent->user_data.name;					\
  }										\
										\
  if (ufuncs->destroy.name)							\
    ufuncs->destroy.name (ufuncs->user_data.name);				\
										\
  if (func)									\
    ufuncs->func.name = func;							\
  else										\
    ufuncs->func.name = ufuncs->parent->func.name;				\
  ufuncs->user_data.name = user_data;						\
  ufuncs->destroy.name = destroy;						\
  return;									\
										\
fail:										\
  if (destroy)									\
    destroy (user_data);							\
}

HB_UNICODE_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_UNICODE_FUNC_IMPLEMENT


#define HB_UNICODE_FUNC_IMPLEMENT(return_type, name)				\
										\
return_type									\
hb_unicode_##name (hb_unicode_funcs_t *ufuncs,					\
		   hb_codepoint_t      unicode)					\
{										\
  return ufuncs->name (unicode);						\
}
HB_UNICODE_FUNCS_IMPLEMENT_CALLBACKS_SIMPLE
#undef HB_UNICODE_FUNC_IMPLEMENT

/**
 * hb_unicode_compose:
 * @ufuncs: The Unicode-functions structure
 * @a: The first Unicode code point to compose
 * @b: The second Unicode code point to compose
 * @ab: (out): The composition of @a, @b
 *
 * Fetches the composition of a sequence of two Unicode
 * code points.
 *
 * Calls the composition function of the specified
 * Unicode-functions structure @ufuncs.
 *
 * Return value: `true` if @a and @b composed, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_unicode_compose (hb_unicode_funcs_t *ufuncs,
		    hb_codepoint_t      a,
		    hb_codepoint_t      b,
		    hb_codepoint_t     *ab)
{
  return ufuncs->compose (a, b, ab);
}

/**
 * hb_unicode_decompose:
 * @ufuncs: The Unicode-functions structure
 * @ab: Unicode code point to decompose
 * @a: (out): The first code point of the decomposition of @ab
 * @b: (out): The second code point of the decomposition of @ab
 *
 * Fetches the decomposition of a Unicode code point. 
 *
 * Calls the decomposition function of the specified
 * Unicode-functions structure @ufuncs.
 *
 * Return value: `true` if @ab was decomposed, `false` otherwise
 *
 * Since: 0.9.2
 **/
hb_bool_t
hb_unicode_decompose (hb_unicode_funcs_t *ufuncs,
		      hb_codepoint_t      ab,
		      hb_codepoint_t     *a,
		      hb_codepoint_t     *b)
{
  return ufuncs->decompose (ab, a, b);
}

#ifndef HB_DISABLE_DEPRECATED
/**
 * hb_unicode_decompose_compatibility:
 * @ufuncs: The Unicode-functions structure
 * @u: Code point to decompose
 * @decomposed: (out): Compatibility decomposition of @u
 *
 * Fetches the compatibility decomposition of a Unicode
 * code point. Deprecated.
 *
 * Return value: length of @decomposed.
 *
 * Since: 0.9.2
 * Deprecated: 2.0.0
 **/
unsigned int
hb_unicode_decompose_compatibility (hb_unicode_funcs_t *ufuncs,
				    hb_codepoint_t      u,
				    hb_codepoint_t     *decomposed)
{
  return ufuncs->decompose_compatibility (u, decomposed);
}
#endif


#ifndef HB_NO_OT_SHAPE
/* See hb-unicode.hh for details. */
const uint8_t
_hb_modified_combining_class[256] =
{
  0, /* HB_UNICODE_COMBINING_CLASS_NOT_REORDERED */
  1, /* HB_UNICODE_COMBINING_CLASS_OVERLAY */
  2, 3, 4, 5, 6,
  7, /* HB_UNICODE_COMBINING_CLASS_NUKTA */
  8, /* HB_UNICODE_COMBINING_CLASS_KANA_VOICING */
  9, /* HB_UNICODE_COMBINING_CLASS_VIRAMA */

  /* Hebrew */
  HB_MODIFIED_COMBINING_CLASS_CCC10,
  HB_MODIFIED_COMBINING_CLASS_CCC11,
  HB_MODIFIED_COMBINING_CLASS_CCC12,
  HB_MODIFIED_COMBINING_CLASS_CCC13,
  HB_MODIFIED_COMBINING_CLASS_CCC14,
  HB_MODIFIED_COMBINING_CLASS_CCC15,
  HB_MODIFIED_COMBINING_CLASS_CCC16,
  HB_MODIFIED_COMBINING_CLASS_CCC17,
  HB_MODIFIED_COMBINING_CLASS_CCC18,
  HB_MODIFIED_COMBINING_CLASS_CCC19,
  HB_MODIFIED_COMBINING_CLASS_CCC20,
  HB_MODIFIED_COMBINING_CLASS_CCC21,
  HB_MODIFIED_COMBINING_CLASS_CCC22,
  HB_MODIFIED_COMBINING_CLASS_CCC23,
  HB_MODIFIED_COMBINING_CLASS_CCC24,
  HB_MODIFIED_COMBINING_CLASS_CCC25,
  HB_MODIFIED_COMBINING_CLASS_CCC26,

  /* Arabic */
  HB_MODIFIED_COMBINING_CLASS_CCC27,
  HB_MODIFIED_COMBINING_CLASS_CCC28,
  HB_MODIFIED_COMBINING_CLASS_CCC29,
  HB_MODIFIED_COMBINING_CLASS_CCC30,
  HB_MODIFIED_COMBINING_CLASS_CCC31,
  HB_MODIFIED_COMBINING_CLASS_CCC32,
  HB_MODIFIED_COMBINING_CLASS_CCC33,
  HB_MODIFIED_COMBINING_CLASS_CCC34,
  HB_MODIFIED_COMBINING_CLASS_CCC35,

  /* Syriac */
  HB_MODIFIED_COMBINING_CLASS_CCC36,

  37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
  60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83,

  /* Telugu */
  HB_MODIFIED_COMBINING_CLASS_CCC84,
  85, 86, 87, 88, 89, 90,
  HB_MODIFIED_COMBINING_CLASS_CCC91,
  92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,

  /* Thai */
  HB_MODIFIED_COMBINING_CLASS_CCC103,
  104, 105, 106,
  HB_MODIFIED_COMBINING_CLASS_CCC107,
  108, 109, 110, 111, 112, 113, 114, 115, 116, 117,

  /* Lao */
  HB_MODIFIED_COMBINING_CLASS_CCC118,
  119, 120, 121,
  HB_MODIFIED_COMBINING_CLASS_CCC122,
  123, 124, 125, 126, 127, 128,

  /* Tibetan */
  HB_MODIFIED_COMBINING_CLASS_CCC129,
  HB_MODIFIED_COMBINING_CLASS_CCC130,
  131,
  HB_MODIFIED_COMBINING_CLASS_CCC132,
  133, 134, 135, 136, 137, 138, 139,


  140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
  150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
  170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
  180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
  190, 191, 192, 193, 194, 195, 196, 197, 198, 199,

  200, /* HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW_LEFT */
  201,
  202, /* HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW */
  203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
  214, /* HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE */
  215,
  216, /* HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE_RIGHT */
  217,
  218, /* HB_UNICODE_COMBINING_CLASS_BELOW_LEFT */
  219,
  220, /* HB_UNICODE_COMBINING_CLASS_BELOW */
  221,
  222, /* HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT */
  223,
  224, /* HB_UNICODE_COMBINING_CLASS_LEFT */
  225,
  226, /* HB_UNICODE_COMBINING_CLASS_RIGHT */
  227,
  228, /* HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT */
  229,
  230, /* HB_UNICODE_COMBINING_CLASS_ABOVE */
  231,
  232, /* HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT */
  233, /* HB_UNICODE_COMBINING_CLASS_DOUBLE_BELOW */
  234, /* HB_UNICODE_COMBINING_CLASS_DOUBLE_ABOVE */
  235, 236, 237, 238, 239,
  240, /* HB_UNICODE_COMBINING_CLASS_IOTA_SUBSCRIPT */
  241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
  255, /* HB_UNICODE_COMBINING_CLASS_INVALID */
};
#endif


/*
 * Emoji
 */
#ifndef HB_NO_EMOJI_SEQUENCES

#include "hb-unicode-emoji-table.hh"

bool
_hb_unicode_is_emoji_Extended_Pictographic (hb_codepoint_t cp)
{
  return _hb_emoji_is_Extended_Pictographic (cp);
}
#endif
