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


  /* All HarfBuzz function declarations used by FreeType, taken    */
  /* from various public HarfBuzz header files.  The wrapper macro */
  /* `HB_EXTERN` is defined in `ft-hb.h`.                          */


/* hb-blob.h */

HB_EXTERN(hb_blob_t *,
hb_blob_create,(const char        *data,
		unsigned int       length,
		hb_memory_mode_t   mode,
		void              *user_data,
		hb_destroy_func_t  destroy))

HB_EXTERN(void,
hb_blob_destroy,(hb_blob_t *blob))


/* hb-buffer.h */

HB_EXTERN(void,
hb_buffer_add_utf8,(hb_buffer_t  *buffer,
		    const char   *text,
		    int           text_length,
		    unsigned int  item_offset,
		    int           item_length))

HB_EXTERN(void,
hb_buffer_clear_contents,(hb_buffer_t *buffer))

HB_EXTERN(hb_buffer_t *,
hb_buffer_create,(void))

HB_EXTERN(void,
hb_buffer_destroy,(hb_buffer_t *buffer))

HB_EXTERN(hb_glyph_info_t *,
hb_buffer_get_glyph_infos,(hb_buffer_t  *buffer,
			   unsigned int *length))

HB_EXTERN(hb_glyph_position_t *,
hb_buffer_get_glyph_positions,(hb_buffer_t  *buffer,
			       unsigned int *length))

HB_EXTERN(unsigned int,
hb_buffer_get_length,(const hb_buffer_t *buffer))

HB_EXTERN(void,
hb_buffer_guess_segment_properties,(hb_buffer_t *buffer))


/* hb-face.h */

HB_EXTERN(hb_face_t *,
hb_face_create,(hb_blob_t    *blob,
		unsigned int  index))

HB_EXTERN(hb_face_t *,
hb_face_create_for_tables,(hb_reference_table_func_t  reference_table_func,
			   void                      *user_data,
			   hb_destroy_func_t          destroy))

HB_EXTERN(void,
hb_face_destroy,(hb_face_t *face))

HB_EXTERN(void,
hb_face_set_index,(hb_face_t    *face,
		   unsigned int  index))

HB_EXTERN(void,
hb_face_set_upem,(hb_face_t    *face,
		  unsigned int  upem))


/* hb-font.h */

HB_EXTERN(hb_font_t *,
hb_font_create,(hb_face_t *face))

HB_EXTERN(void,
hb_font_destroy,(hb_font_t *font))

HB_EXTERN(hb_face_t *,
hb_font_get_face,(hb_font_t *font))

HB_EXTERN(void,
hb_font_set_scale,(hb_font_t *font,
		   int x_scale,
		   int y_scale))


/* hb-ot-layout.h */

HB_EXTERN(void,
hb_ot_layout_collect_lookups,(hb_face_t      *face,
			      hb_tag_t        table_tag,
			      const hb_tag_t *scripts,
			      const hb_tag_t *languages,
			      const hb_tag_t *features,
			      hb_set_t       *lookup_indexes /* OUT */))

HB_EXTERN(void,
hb_ot_layout_lookup_collect_glyphs,(hb_face_t    *face,
				    hb_tag_t      table_tag,
				    unsigned int  lookup_index,
				    hb_set_t     *glyphs_before, /* OUT.  May be NULL */
				    hb_set_t     *glyphs_input,  /* OUT.  May be NULL */
				    hb_set_t     *glyphs_after,  /* OUT.  May be NULL */
				    hb_set_t     *glyphs_output  /* OUT.  May be NULL */))

HB_EXTERN(hb_bool_t,
hb_ot_layout_lookup_would_substitute,(hb_face_t            *face,
				      unsigned int          lookup_index,
				      const hb_codepoint_t *glyphs,
				      unsigned int          glyphs_length,
				      hb_bool_t             zero_context))

HB_EXTERN(void,
hb_ot_tags_from_script_and_language,(hb_script_t   script,
				     hb_language_t language,
				     unsigned int *script_count /* IN/OUT */,
				     hb_tag_t     *script_tags /* OUT */,
				     unsigned int *language_count /* IN/OUT */,
				     hb_tag_t     *language_tags /* OUT */))


/* hb-set.h */

HB_EXTERN(void,
hb_set_add,(hb_set_t       *set,
	    hb_codepoint_t  codepoint))

HB_EXTERN(void,
hb_set_clear,(hb_set_t *set))

HB_EXTERN(hb_set_t *,
hb_set_create,(void))

HB_EXTERN(void,
hb_set_destroy,(hb_set_t *set))

HB_EXTERN(void,
hb_set_del,(hb_set_t       *set,
	    hb_codepoint_t  codepoint))

HB_EXTERN(hb_bool_t,
hb_set_has,(const hb_set_t *set,
	    hb_codepoint_t  codepoint))

HB_EXTERN(hb_bool_t,
hb_set_is_empty,(const hb_set_t *set))

HB_EXTERN(hb_bool_t,
hb_set_next,(const hb_set_t *set,
	     hb_codepoint_t *codepoint))

HB_EXTERN(void,
hb_set_subtract,(hb_set_t       *set,
		 const hb_set_t *other))


/* hb-shape.h */

HB_EXTERN(void,
hb_shape,(hb_font_t           *font,
	  hb_buffer_t         *buffer,
	  const hb_feature_t  *features,
	  unsigned int         num_features))

HB_EXTERN(hb_bool_t,
hb_version_atleast,(unsigned int major,
		    unsigned int minor,
		    unsigned int micro))


/* END */
