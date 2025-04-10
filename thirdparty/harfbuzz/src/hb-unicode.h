/*
 * Copyright © 2009  Red Hat, Inc.
 * Copyright © 2011  Codethink Limited
 * Copyright © 2011,2012  Google, Inc.
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

#if !defined(HB_H_IN) && !defined(HB_NO_SINGLE_HEADER_ERROR)
#error "Include <hb.h> instead."
#endif

#ifndef HB_UNICODE_H
#define HB_UNICODE_H

#include "hb-common.h"

HB_BEGIN_DECLS


/**
 * HB_UNICODE_MAX:
 *
 * Maximum valid Unicode code point.
 *
 * Since: 1.9.0
 **/
#define HB_UNICODE_MAX 0x10FFFFu


/**
 * hb_unicode_general_category_t:
 * @HB_UNICODE_GENERAL_CATEGORY_CONTROL:              [Cc]
 * @HB_UNICODE_GENERAL_CATEGORY_FORMAT:		      [Cf]
 * @HB_UNICODE_GENERAL_CATEGORY_UNASSIGNED:	      [Cn]
 * @HB_UNICODE_GENERAL_CATEGORY_PRIVATE_USE:	      [Co]
 * @HB_UNICODE_GENERAL_CATEGORY_SURROGATE:	      [Cs]
 * @HB_UNICODE_GENERAL_CATEGORY_LOWERCASE_LETTER:     [Ll]
 * @HB_UNICODE_GENERAL_CATEGORY_MODIFIER_LETTER:      [Lm]
 * @HB_UNICODE_GENERAL_CATEGORY_OTHER_LETTER:	      [Lo]
 * @HB_UNICODE_GENERAL_CATEGORY_TITLECASE_LETTER:     [Lt]
 * @HB_UNICODE_GENERAL_CATEGORY_UPPERCASE_LETTER:     [Lu]
 * @HB_UNICODE_GENERAL_CATEGORY_SPACING_MARK:	      [Mc]
 * @HB_UNICODE_GENERAL_CATEGORY_ENCLOSING_MARK:	      [Me]
 * @HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK:     [Mn]
 * @HB_UNICODE_GENERAL_CATEGORY_DECIMAL_NUMBER:	      [Nd]
 * @HB_UNICODE_GENERAL_CATEGORY_LETTER_NUMBER:	      [Nl]
 * @HB_UNICODE_GENERAL_CATEGORY_OTHER_NUMBER:	      [No]
 * @HB_UNICODE_GENERAL_CATEGORY_CONNECT_PUNCTUATION:  [Pc]
 * @HB_UNICODE_GENERAL_CATEGORY_DASH_PUNCTUATION:     [Pd]
 * @HB_UNICODE_GENERAL_CATEGORY_CLOSE_PUNCTUATION:    [Pe]
 * @HB_UNICODE_GENERAL_CATEGORY_FINAL_PUNCTUATION:    [Pf]
 * @HB_UNICODE_GENERAL_CATEGORY_INITIAL_PUNCTUATION:  [Pi]
 * @HB_UNICODE_GENERAL_CATEGORY_OTHER_PUNCTUATION:    [Po]
 * @HB_UNICODE_GENERAL_CATEGORY_OPEN_PUNCTUATION:     [Ps]
 * @HB_UNICODE_GENERAL_CATEGORY_CURRENCY_SYMBOL:      [Sc]
 * @HB_UNICODE_GENERAL_CATEGORY_MODIFIER_SYMBOL:      [Sk]
 * @HB_UNICODE_GENERAL_CATEGORY_MATH_SYMBOL:	      [Sm]
 * @HB_UNICODE_GENERAL_CATEGORY_OTHER_SYMBOL:	      [So]
 * @HB_UNICODE_GENERAL_CATEGORY_LINE_SEPARATOR:	      [Zl]
 * @HB_UNICODE_GENERAL_CATEGORY_PARAGRAPH_SEPARATOR:  [Zp]
 * @HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR:      [Zs]
 *
 * Data type for the "General_Category" (gc) property from
 * the Unicode Character Database.
 **/

/* Unicode Character Database property: General_Category (gc) */
typedef enum
{
  HB_UNICODE_GENERAL_CATEGORY_CONTROL,			/* Cc */
  HB_UNICODE_GENERAL_CATEGORY_FORMAT,			/* Cf */
  HB_UNICODE_GENERAL_CATEGORY_UNASSIGNED,		/* Cn */
  HB_UNICODE_GENERAL_CATEGORY_PRIVATE_USE,		/* Co */
  HB_UNICODE_GENERAL_CATEGORY_SURROGATE,		/* Cs */
  HB_UNICODE_GENERAL_CATEGORY_LOWERCASE_LETTER,		/* Ll */
  HB_UNICODE_GENERAL_CATEGORY_MODIFIER_LETTER,		/* Lm */
  HB_UNICODE_GENERAL_CATEGORY_OTHER_LETTER,		/* Lo */
  HB_UNICODE_GENERAL_CATEGORY_TITLECASE_LETTER,		/* Lt */
  HB_UNICODE_GENERAL_CATEGORY_UPPERCASE_LETTER,		/* Lu */
  HB_UNICODE_GENERAL_CATEGORY_SPACING_MARK,		/* Mc */
  HB_UNICODE_GENERAL_CATEGORY_ENCLOSING_MARK,		/* Me */
  HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK,		/* Mn */
  HB_UNICODE_GENERAL_CATEGORY_DECIMAL_NUMBER,		/* Nd */
  HB_UNICODE_GENERAL_CATEGORY_LETTER_NUMBER,		/* Nl */
  HB_UNICODE_GENERAL_CATEGORY_OTHER_NUMBER,		/* No */
  HB_UNICODE_GENERAL_CATEGORY_CONNECT_PUNCTUATION,	/* Pc */
  HB_UNICODE_GENERAL_CATEGORY_DASH_PUNCTUATION,		/* Pd */
  HB_UNICODE_GENERAL_CATEGORY_CLOSE_PUNCTUATION,	/* Pe */
  HB_UNICODE_GENERAL_CATEGORY_FINAL_PUNCTUATION,	/* Pf */
  HB_UNICODE_GENERAL_CATEGORY_INITIAL_PUNCTUATION,	/* Pi */
  HB_UNICODE_GENERAL_CATEGORY_OTHER_PUNCTUATION,	/* Po */
  HB_UNICODE_GENERAL_CATEGORY_OPEN_PUNCTUATION,		/* Ps */
  HB_UNICODE_GENERAL_CATEGORY_CURRENCY_SYMBOL,		/* Sc */
  HB_UNICODE_GENERAL_CATEGORY_MODIFIER_SYMBOL,		/* Sk */
  HB_UNICODE_GENERAL_CATEGORY_MATH_SYMBOL,		/* Sm */
  HB_UNICODE_GENERAL_CATEGORY_OTHER_SYMBOL,		/* So */
  HB_UNICODE_GENERAL_CATEGORY_LINE_SEPARATOR,		/* Zl */
  HB_UNICODE_GENERAL_CATEGORY_PARAGRAPH_SEPARATOR,	/* Zp */
  HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR		/* Zs */
} hb_unicode_general_category_t;

/**
 * hb_unicode_combining_class_t:
 * @HB_UNICODE_COMBINING_CLASS_NOT_REORDERED: Spacing and enclosing marks; also many vowel and consonant signs, even if nonspacing
 * @HB_UNICODE_COMBINING_CLASS_OVERLAY: Marks which overlay a base letter or symbol
 * @HB_UNICODE_COMBINING_CLASS_NUKTA: Diacritic nukta marks in Brahmi-derived scripts
 * @HB_UNICODE_COMBINING_CLASS_KANA_VOICING: Hiragana/Katakana voicing marks
 * @HB_UNICODE_COMBINING_CLASS_VIRAMA: Viramas
 * @HB_UNICODE_COMBINING_CLASS_CCC10: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC11: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC12: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC13: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC14: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC15: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC16: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC17: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC18: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC19: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC20: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC21: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC22: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC23: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC24: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC25: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC26: [Hebrew]
 * @HB_UNICODE_COMBINING_CLASS_CCC27: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC28: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC29: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC30: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC31: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC32: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC33: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC34: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC35: [Arabic]
 * @HB_UNICODE_COMBINING_CLASS_CCC36: [Syriac]
 * @HB_UNICODE_COMBINING_CLASS_CCC84: [Telugu]
 * @HB_UNICODE_COMBINING_CLASS_CCC91: [Telugu]
 * @HB_UNICODE_COMBINING_CLASS_CCC103: [Thai]
 * @HB_UNICODE_COMBINING_CLASS_CCC107: [Thai]
 * @HB_UNICODE_COMBINING_CLASS_CCC118: [Lao]
 * @HB_UNICODE_COMBINING_CLASS_CCC122: [Lao]
 * @HB_UNICODE_COMBINING_CLASS_CCC129: [Tibetan]
 * @HB_UNICODE_COMBINING_CLASS_CCC130: [Tibetan]
 * @HB_UNICODE_COMBINING_CLASS_CCC132: [Tibetan] Since: 7.2.0
 * @HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW_LEFT: Marks attached at the bottom left
 * @HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW: Marks attached directly below
 * @HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE: Marks attached directly above
 * @HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE_RIGHT: Marks attached at the top right
 * @HB_UNICODE_COMBINING_CLASS_BELOW_LEFT: Distinct marks at the bottom left
 * @HB_UNICODE_COMBINING_CLASS_BELOW: Distinct marks directly below
 * @HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT: Distinct marks at the bottom right
 * @HB_UNICODE_COMBINING_CLASS_LEFT: Distinct marks to the left
 * @HB_UNICODE_COMBINING_CLASS_RIGHT: Distinct marks to the right
 * @HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT: Distinct marks at the top left
 * @HB_UNICODE_COMBINING_CLASS_ABOVE: Distinct marks directly above
 * @HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT: Distinct marks at the top right
 * @HB_UNICODE_COMBINING_CLASS_DOUBLE_BELOW: Distinct marks subtending two bases
 * @HB_UNICODE_COMBINING_CLASS_DOUBLE_ABOVE: Distinct marks extending above two bases
 * @HB_UNICODE_COMBINING_CLASS_IOTA_SUBSCRIPT: Greek iota subscript only
 * @HB_UNICODE_COMBINING_CLASS_INVALID: Invalid combining class
 *
 * Data type for the Canonical_Combining_Class (ccc) property
 * from the Unicode Character Database.
 *
 * <note>Note: newer versions of Unicode may add new values.
 * Client programs should be ready to handle any value in the 0..254 range
 * being returned from hb_unicode_combining_class().</note>
 *
 **/
typedef enum
{
  HB_UNICODE_COMBINING_CLASS_NOT_REORDERED	= 0,
  HB_UNICODE_COMBINING_CLASS_OVERLAY		= 1,
  HB_UNICODE_COMBINING_CLASS_NUKTA		= 7,
  HB_UNICODE_COMBINING_CLASS_KANA_VOICING	= 8,
  HB_UNICODE_COMBINING_CLASS_VIRAMA		= 9,

  /* Hebrew */
  HB_UNICODE_COMBINING_CLASS_CCC10	=  10,
  HB_UNICODE_COMBINING_CLASS_CCC11	=  11,
  HB_UNICODE_COMBINING_CLASS_CCC12	=  12,
  HB_UNICODE_COMBINING_CLASS_CCC13	=  13,
  HB_UNICODE_COMBINING_CLASS_CCC14	=  14,
  HB_UNICODE_COMBINING_CLASS_CCC15	=  15,
  HB_UNICODE_COMBINING_CLASS_CCC16	=  16,
  HB_UNICODE_COMBINING_CLASS_CCC17	=  17,
  HB_UNICODE_COMBINING_CLASS_CCC18	=  18,
  HB_UNICODE_COMBINING_CLASS_CCC19	=  19,
  HB_UNICODE_COMBINING_CLASS_CCC20	=  20,
  HB_UNICODE_COMBINING_CLASS_CCC21	=  21,
  HB_UNICODE_COMBINING_CLASS_CCC22	=  22,
  HB_UNICODE_COMBINING_CLASS_CCC23	=  23,
  HB_UNICODE_COMBINING_CLASS_CCC24	=  24,
  HB_UNICODE_COMBINING_CLASS_CCC25	=  25,
  HB_UNICODE_COMBINING_CLASS_CCC26	=  26,

  /* Arabic */
  HB_UNICODE_COMBINING_CLASS_CCC27	=  27,
  HB_UNICODE_COMBINING_CLASS_CCC28	=  28,
  HB_UNICODE_COMBINING_CLASS_CCC29	=  29,
  HB_UNICODE_COMBINING_CLASS_CCC30	=  30,
  HB_UNICODE_COMBINING_CLASS_CCC31	=  31,
  HB_UNICODE_COMBINING_CLASS_CCC32	=  32,
  HB_UNICODE_COMBINING_CLASS_CCC33	=  33,
  HB_UNICODE_COMBINING_CLASS_CCC34	=  34,
  HB_UNICODE_COMBINING_CLASS_CCC35	=  35,

  /* Syriac */
  HB_UNICODE_COMBINING_CLASS_CCC36	=  36,

  /* Telugu */
  HB_UNICODE_COMBINING_CLASS_CCC84	=  84,
  HB_UNICODE_COMBINING_CLASS_CCC91	=  91,

  /* Thai */
  HB_UNICODE_COMBINING_CLASS_CCC103	= 103,
  HB_UNICODE_COMBINING_CLASS_CCC107	= 107,

  /* Lao */
  HB_UNICODE_COMBINING_CLASS_CCC118	= 118,
  HB_UNICODE_COMBINING_CLASS_CCC122	= 122,

  /* Tibetan */
  HB_UNICODE_COMBINING_CLASS_CCC129	= 129,
  HB_UNICODE_COMBINING_CLASS_CCC130	= 130,
  HB_UNICODE_COMBINING_CLASS_CCC132	= 132,


  HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW_LEFT	= 200,
  HB_UNICODE_COMBINING_CLASS_ATTACHED_BELOW		= 202,
  HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE		= 214,
  HB_UNICODE_COMBINING_CLASS_ATTACHED_ABOVE_RIGHT	= 216,
  HB_UNICODE_COMBINING_CLASS_BELOW_LEFT			= 218,
  HB_UNICODE_COMBINING_CLASS_BELOW			= 220,
  HB_UNICODE_COMBINING_CLASS_BELOW_RIGHT		= 222,
  HB_UNICODE_COMBINING_CLASS_LEFT			= 224,
  HB_UNICODE_COMBINING_CLASS_RIGHT			= 226,
  HB_UNICODE_COMBINING_CLASS_ABOVE_LEFT			= 228,
  HB_UNICODE_COMBINING_CLASS_ABOVE			= 230,
  HB_UNICODE_COMBINING_CLASS_ABOVE_RIGHT		= 232,
  HB_UNICODE_COMBINING_CLASS_DOUBLE_BELOW		= 233,
  HB_UNICODE_COMBINING_CLASS_DOUBLE_ABOVE		= 234,

  HB_UNICODE_COMBINING_CLASS_IOTA_SUBSCRIPT		= 240,

  HB_UNICODE_COMBINING_CLASS_INVALID	= 255
} hb_unicode_combining_class_t;


/*
 * hb_unicode_funcs_t
 */

/**
 * hb_unicode_funcs_t:
 *
 * Data type containing a set of virtual methods used for
 * accessing various Unicode character properties.
 *
 * HarfBuzz provides a default function for each of the
 * methods in #hb_unicode_funcs_t. Client programs can implement
 * their own replacements for the individual Unicode functions, as
 * needed, and replace the default by calling the setter for a
 * method.
 **/
typedef struct hb_unicode_funcs_t hb_unicode_funcs_t;


/*
 * just give me the best implementation you've got there.
 */
HB_EXTERN hb_unicode_funcs_t *
hb_unicode_funcs_get_default (void);


HB_EXTERN hb_unicode_funcs_t *
hb_unicode_funcs_create (hb_unicode_funcs_t *parent);

HB_EXTERN hb_unicode_funcs_t *
hb_unicode_funcs_get_empty (void);

HB_EXTERN hb_unicode_funcs_t *
hb_unicode_funcs_reference (hb_unicode_funcs_t *ufuncs);

HB_EXTERN void
hb_unicode_funcs_destroy (hb_unicode_funcs_t *ufuncs);

HB_EXTERN hb_bool_t
hb_unicode_funcs_set_user_data (hb_unicode_funcs_t *ufuncs,
				hb_user_data_key_t *key,
				void *              data,
				hb_destroy_func_t   destroy,
				hb_bool_t           replace);


HB_EXTERN void *
hb_unicode_funcs_get_user_data (const hb_unicode_funcs_t *ufuncs,
				hb_user_data_key_t       *key);


HB_EXTERN void
hb_unicode_funcs_make_immutable (hb_unicode_funcs_t *ufuncs);

HB_EXTERN hb_bool_t
hb_unicode_funcs_is_immutable (hb_unicode_funcs_t *ufuncs);

HB_EXTERN hb_unicode_funcs_t *
hb_unicode_funcs_get_parent (hb_unicode_funcs_t *ufuncs);


/*
 * funcs
 */

/* typedefs */

/**
 * hb_unicode_combining_class_func_t:
 * @ufuncs: A Unicode-functions structure
 * @unicode: The code point to query
 * @user_data: User data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should retrieve the Canonical Combining Class (ccc)
 * property for a specified Unicode code point. 
 *
 * Return value: The #hb_unicode_combining_class_t of @unicode
 * 
 **/
typedef hb_unicode_combining_class_t	(*hb_unicode_combining_class_func_t)	(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      unicode,
										 void               *user_data);

/**
 * hb_unicode_general_category_func_t:
 * @ufuncs: A Unicode-functions structure
 * @unicode: The code point to query
 * @user_data: User data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should retrieve the General Category property for
 * a specified Unicode code point.
 * 
 * Return value: The #hb_unicode_general_category_t of @unicode
 *
 **/
typedef hb_unicode_general_category_t	(*hb_unicode_general_category_func_t)	(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      unicode,
										 void               *user_data);

/**
 * hb_unicode_mirroring_func_t:
 * @ufuncs: A Unicode-functions structure
 * @unicode: The code point to query
 * @user_data: User data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should retrieve the Bi-Directional Mirroring Glyph
 * code point for a specified Unicode code point.
 *
 * <note>Note: If a code point does not have a specified
 * Bi-Directional Mirroring Glyph defined, the method should
 * return the original code point.</note>
 * 
 * Return value: The #hb_codepoint_t of the Mirroring Glyph for @unicode
 *
 **/
typedef hb_codepoint_t			(*hb_unicode_mirroring_func_t)		(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      unicode,
										 void               *user_data);

/**
 * hb_unicode_script_func_t:
 * @ufuncs: A Unicode-functions structure
 * @unicode: The code point to query
 * @user_data: User data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should retrieve the Script property for a 
 * specified Unicode code point.
 *
 * Return value: The #hb_script_t of @unicode
 * 
 **/
typedef hb_script_t			(*hb_unicode_script_func_t)		(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      unicode,
										 void               *user_data);

/**
 * hb_unicode_compose_func_t:
 * @ufuncs: A Unicode-functions structure
 * @a: The first code point to compose
 * @b: The second code point to compose
 * @ab: (out): The composed code point
 * @user_data: user data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should compose a sequence of two input Unicode code
 * points by canonical equivalence, returning the composed code
 * point in a #hb_codepoint_t output parameter (if successful).
 * The method must return an #hb_bool_t indicating the success
 * of the composition.
 * 
 * Return value: `true` is @a,@b composed, `false` otherwise
 *
 **/
typedef hb_bool_t			(*hb_unicode_compose_func_t)		(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      a,
										 hb_codepoint_t      b,
										 hb_codepoint_t     *ab,
										 void               *user_data);

/**
 * hb_unicode_decompose_func_t:
 * @ufuncs: A Unicode-functions structure
 * @ab: The code point to decompose
 * @a: (out): The first decomposed code point
 * @b: (out): The second decomposed code point
 * @user_data: user data pointer passed by the caller
 *
 * A virtual method for the #hb_unicode_funcs_t structure.
 *
 * This method should decompose an input Unicode code point,
 * returning the two decomposed code points in #hb_codepoint_t
 * output parameters (if successful). The method must return an
 * #hb_bool_t indicating the success of the composition.
 * 
 * Return value: `true` if @ab decomposed, `false` otherwise
 *
 **/
typedef hb_bool_t			(*hb_unicode_decompose_func_t)		(hb_unicode_funcs_t *ufuncs,
										 hb_codepoint_t      ab,
										 hb_codepoint_t     *a,
										 hb_codepoint_t     *b,
										 void               *user_data);

/* func setters */

/**
 * hb_unicode_funcs_set_combining_class_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_combining_class_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_combining_class_func (hb_unicode_funcs_t *ufuncs,
					   hb_unicode_combining_class_func_t func,
					   void *user_data, hb_destroy_func_t destroy);

/**
 * hb_unicode_funcs_set_general_category_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_general_category_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_general_category_func (hb_unicode_funcs_t *ufuncs,
					    hb_unicode_general_category_func_t func,
					    void *user_data, hb_destroy_func_t destroy);

/**
 * hb_unicode_funcs_set_mirroring_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_mirroring_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_mirroring_func (hb_unicode_funcs_t *ufuncs,
				     hb_unicode_mirroring_func_t func,
				     void *user_data, hb_destroy_func_t destroy);

/**
 * hb_unicode_funcs_set_script_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_script_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_script_func (hb_unicode_funcs_t *ufuncs,
				  hb_unicode_script_func_t func,
				  void *user_data, hb_destroy_func_t destroy);

/**
 * hb_unicode_funcs_set_compose_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_compose_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_compose_func (hb_unicode_funcs_t *ufuncs,
				   hb_unicode_compose_func_t func,
				   void *user_data, hb_destroy_func_t destroy);

/**
 * hb_unicode_funcs_set_decompose_func:
 * @ufuncs: A Unicode-functions structure
 * @func: (closure user_data) (destroy destroy) (scope notified): The callback function to assign
 * @user_data: Data to pass to @func
 * @destroy: (nullable): The function to call when @user_data is not needed anymore
 *
 * Sets the implementation function for #hb_unicode_decompose_func_t.
 *
 * Since: 0.9.2
 **/
HB_EXTERN void
hb_unicode_funcs_set_decompose_func (hb_unicode_funcs_t *ufuncs,
				     hb_unicode_decompose_func_t func,
				     void *user_data, hb_destroy_func_t destroy);

/* accessors */

/**
 * hb_unicode_combining_class:
 * @ufuncs: The Unicode-functions structure
 * @unicode: The code point to query
 *
 * Retrieves the Canonical Combining Class (ccc) property
 * of code point @unicode.
 *
 * Return value: The #hb_unicode_combining_class_t of @unicode
 *
 * Since: 0.9.2
 **/
HB_EXTERN hb_unicode_combining_class_t
hb_unicode_combining_class (hb_unicode_funcs_t *ufuncs,
			    hb_codepoint_t unicode);

/**
 * hb_unicode_general_category:
 * @ufuncs: The Unicode-functions structure
 * @unicode: The code point to query
 *
 * Retrieves the General Category (gc) property
 * of code point @unicode.
 *
 * Return value: The #hb_unicode_general_category_t of @unicode
 *
 * Since: 0.9.2
 **/
HB_EXTERN hb_unicode_general_category_t
hb_unicode_general_category (hb_unicode_funcs_t *ufuncs,
			     hb_codepoint_t unicode);

/**
 * hb_unicode_mirroring:
 * @ufuncs: The Unicode-functions structure
 * @unicode: The code point to query
 *
 * Retrieves the Bi-directional Mirroring Glyph code
 * point defined for code point @unicode.
 *
 * Return value: The #hb_codepoint_t of the Mirroring Glyph for @unicode
 *
 * Since: 0.9.2
 **/
HB_EXTERN hb_codepoint_t
hb_unicode_mirroring (hb_unicode_funcs_t *ufuncs,
		      hb_codepoint_t unicode);

/**
 * hb_unicode_script:
 * @ufuncs: The Unicode-functions structure
 * @unicode: The code point to query
 *
 * Retrieves the #hb_script_t script to which code
 * point @unicode belongs.
 *
 * Return value: The #hb_script_t of @unicode
 *
 * Since: 0.9.2
 **/
HB_EXTERN hb_script_t
hb_unicode_script (hb_unicode_funcs_t *ufuncs,
		   hb_codepoint_t unicode);

HB_EXTERN hb_bool_t
hb_unicode_compose (hb_unicode_funcs_t *ufuncs,
		    hb_codepoint_t      a,
		    hb_codepoint_t      b,
		    hb_codepoint_t     *ab);

HB_EXTERN hb_bool_t
hb_unicode_decompose (hb_unicode_funcs_t *ufuncs,
		      hb_codepoint_t      ab,
		      hb_codepoint_t     *a,
		      hb_codepoint_t     *b);

HB_END_DECLS

#endif /* HB_UNICODE_H */
