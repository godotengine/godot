/*
 * Copyright © 2019-2020  Ebrahim Byagowi
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

#ifndef HB_NO_DRAW
#ifdef HB_EXPERIMENTAL_API

#include "hb-draw.hh"
#include "hb-ot.h"
#include "hb-ot-glyf-table.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-ot-cff2-table.hh"

/**
 * hb_draw_funcs_set_move_to_func:
 * @funcs: draw functions object
 * @move_to: move-to callback
 *
 * Sets move-to callback to the draw functions object.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_set_move_to_func (hb_draw_funcs_t        *funcs,
				hb_draw_move_to_func_t  move_to)
{
  if (unlikely (hb_object_is_immutable (funcs))) return;
  funcs->move_to = move_to;
}

/**
 * hb_draw_funcs_set_line_to_func:
 * @funcs: draw functions object
 * @line_to: line-to callback
 *
 * Sets line-to callback to the draw functions object.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_set_line_to_func (hb_draw_funcs_t        *funcs,
				hb_draw_line_to_func_t  line_to)
{
  if (unlikely (hb_object_is_immutable (funcs))) return;
  funcs->line_to = line_to;
}

/**
 * hb_draw_funcs_set_quadratic_to_func:
 * @funcs: draw functions object
 * @move_to: quadratic-to callback
 *
 * Sets quadratic-to callback to the draw functions object.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_set_quadratic_to_func (hb_draw_funcs_t             *funcs,
				     hb_draw_quadratic_to_func_t  quadratic_to)
{
  if (unlikely (hb_object_is_immutable (funcs))) return;
  funcs->quadratic_to = quadratic_to;
  funcs->is_quadratic_to_set = true;
}

/**
 * hb_draw_funcs_set_cubic_to_func:
 * @funcs: draw functions
 * @cubic_to: cubic-to callback
 *
 * Sets cubic-to callback to the draw functions object.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_set_cubic_to_func (hb_draw_funcs_t         *funcs,
				 hb_draw_cubic_to_func_t  cubic_to)
{
  if (unlikely (hb_object_is_immutable (funcs))) return;
  funcs->cubic_to = cubic_to;
}

/**
 * hb_draw_funcs_set_close_path_func:
 * @funcs: draw functions object
 * @close_path: close-path callback
 *
 * Sets close-path callback to the draw functions object.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_set_close_path_func (hb_draw_funcs_t           *funcs,
				   hb_draw_close_path_func_t  close_path)
{
  if (unlikely (hb_object_is_immutable (funcs))) return;
  funcs->close_path = close_path;
}

static void
_move_to_nil (hb_position_t to_x HB_UNUSED, hb_position_t to_y HB_UNUSED, void *user_data HB_UNUSED) {}

static void
_line_to_nil (hb_position_t to_x HB_UNUSED, hb_position_t to_y HB_UNUSED, void *user_data HB_UNUSED) {}

static void
_quadratic_to_nil (hb_position_t control_x HB_UNUSED, hb_position_t control_y HB_UNUSED,
		   hb_position_t to_x HB_UNUSED, hb_position_t to_y HB_UNUSED,
		   void *user_data HB_UNUSED) {}

static void
_cubic_to_nil (hb_position_t control1_x HB_UNUSED, hb_position_t control1_y HB_UNUSED,
	       hb_position_t control2_x HB_UNUSED, hb_position_t control2_y HB_UNUSED,
	       hb_position_t to_x HB_UNUSED, hb_position_t to_y HB_UNUSED,
	       void *user_data HB_UNUSED) {}

static void
_close_path_nil (void *user_data HB_UNUSED) {}

/**
 * hb_draw_funcs_create:
 *
 * Creates a new draw callbacks object.
 *
 * Since: EXPERIMENTAL
 **/
hb_draw_funcs_t *
hb_draw_funcs_create ()
{
  hb_draw_funcs_t *funcs;
  if (unlikely (!(funcs = hb_object_create<hb_draw_funcs_t> ())))
    return const_cast<hb_draw_funcs_t *> (&Null (hb_draw_funcs_t));

  funcs->move_to = (hb_draw_move_to_func_t) _move_to_nil;
  funcs->line_to = (hb_draw_line_to_func_t) _line_to_nil;
  funcs->quadratic_to = (hb_draw_quadratic_to_func_t) _quadratic_to_nil;
  funcs->is_quadratic_to_set = false;
  funcs->cubic_to = (hb_draw_cubic_to_func_t) _cubic_to_nil;
  funcs->close_path = (hb_draw_close_path_func_t) _close_path_nil;
  return funcs;
}

/**
 * hb_draw_funcs_reference:
 * @funcs: draw functions
 *
 * Add to callbacks object refcount.
 *
 * Returns: The same object.
 * Since: EXPERIMENTAL
 **/
hb_draw_funcs_t *
hb_draw_funcs_reference (hb_draw_funcs_t *funcs)
{
  return hb_object_reference (funcs);
}

/**
 * hb_draw_funcs_destroy:
 * @funcs: draw functions
 *
 * Decreases refcount of callbacks object and deletes the object if it reaches
 * to zero.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_destroy (hb_draw_funcs_t *funcs)
{
  if (!hb_object_destroy (funcs)) return;

  free (funcs);
}

/**
 * hb_draw_funcs_make_immutable:
 * @funcs: draw functions
 *
 * Makes funcs object immutable.
 *
 * Since: EXPERIMENTAL
 **/
void
hb_draw_funcs_make_immutable (hb_draw_funcs_t *funcs)
{
  if (hb_object_is_immutable (funcs))
    return;

  hb_object_make_immutable (funcs);
}

/**
 * hb_draw_funcs_is_immutable:
 * @funcs: draw functions
 *
 * Checks whether funcs is immutable.
 *
 * Returns: If is immutable.
 * Since: EXPERIMENTAL
 **/
hb_bool_t
hb_draw_funcs_is_immutable (hb_draw_funcs_t *funcs)
{
  return hb_object_is_immutable (funcs);
}

/**
 * hb_font_draw_glyph:
 * @font: a font object
 * @glyph: a glyph id
 * @funcs: draw callbacks object
 * @user_data: parameter you like be passed to the callbacks when are called
 *
 * Draw a glyph.
 *
 * Returns: Whether the font had the glyph and the operation completed successfully.
 * Since: EXPERIMENTAL
 **/
hb_bool_t
hb_font_draw_glyph (hb_font_t *font, hb_codepoint_t glyph,
		    const hb_draw_funcs_t *funcs,
		    void *user_data)
{
  if (unlikely (funcs == &Null (hb_draw_funcs_t) ||
		glyph >= font->face->get_num_glyphs ()))
    return false;

  draw_helper_t draw_helper (funcs, user_data);
  if (font->face->table.glyf->get_path (font, glyph, draw_helper)) return true;
#ifndef HB_NO_CFF
  if (font->face->table.cff1->get_path (font, glyph, draw_helper)) return true;
  if (font->face->table.cff2->get_path (font, glyph, draw_helper)) return true;
#endif

  return false;
}

#endif
#endif
