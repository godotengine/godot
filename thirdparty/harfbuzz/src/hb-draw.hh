/*
 * Copyright © 2020  Ebrahim Byagowi
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

#ifndef HB_DRAW_HH
#define HB_DRAW_HH

#include "hb.hh"

#ifdef HB_EXPERIMENTAL_API
struct hb_draw_funcs_t
{
  hb_object_header_t header;

  hb_draw_move_to_func_t move_to;
  hb_draw_line_to_func_t line_to;
  hb_draw_quadratic_to_func_t quadratic_to;
  bool is_quadratic_to_set;
  hb_draw_cubic_to_func_t cubic_to;
  hb_draw_close_path_func_t close_path;
};

struct draw_helper_t
{
  draw_helper_t (const hb_draw_funcs_t *funcs_, void *user_data_)
  {
    funcs = funcs_;
    user_data = user_data_;
    path_open = false;
    path_start_x = current_x = path_start_y = current_y = 0;
  }
  ~draw_helper_t () { end_path (); }

  void move_to (hb_position_t x, hb_position_t y)
  {
    if (path_open) end_path ();
    current_x = path_start_x = x;
    current_y = path_start_y = y;
  }

  void line_to (hb_position_t x, hb_position_t y)
  {
    if (equal_to_current (x, y)) return;
    if (!path_open) start_path ();
    funcs->line_to (x, y, user_data);
    current_x = x;
    current_y = y;
  }

  void
  quadratic_to (hb_position_t control_x, hb_position_t control_y,
		hb_position_t to_x, hb_position_t to_y)
  {
    if (equal_to_current (control_x, control_y) && equal_to_current (to_x, to_y))
      return;
    if (!path_open) start_path ();
    if (funcs->is_quadratic_to_set)
      funcs->quadratic_to (control_x, control_y, to_x, to_y, user_data);
    else
      funcs->cubic_to (roundf ((current_x + 2.f * control_x) / 3.f),
		       roundf ((current_y + 2.f * control_y) / 3.f),
		       roundf ((to_x + 2.f * control_x) / 3.f),
		       roundf ((to_y + 2.f * control_y) / 3.f),
		       to_x, to_y, user_data);
    current_x = to_x;
    current_y = to_y;
  }

  void
  cubic_to (hb_position_t control1_x, hb_position_t control1_y,
	    hb_position_t control2_x, hb_position_t control2_y,
	    hb_position_t to_x, hb_position_t to_y)
  {
    if (equal_to_current (control1_x, control1_y) &&
	equal_to_current (control2_x, control2_y) &&
	equal_to_current (to_x, to_y))
      return;
    if (!path_open) start_path ();
    funcs->cubic_to (control1_x, control1_y, control2_x, control2_y, to_x, to_y, user_data);
    current_x = to_x;
    current_y = to_y;
  }

  void end_path ()
  {
    if (path_open)
    {
      if ((path_start_x != current_x) || (path_start_y != current_y))
	funcs->line_to (path_start_x, path_start_y, user_data);
      funcs->close_path (user_data);
    }
    path_open = false;
    path_start_x = current_x = path_start_y = current_y = 0;
  }

  protected:
  bool equal_to_current (hb_position_t x, hb_position_t y)
  { return current_x == x && current_y == y; }

  void start_path ()
  {
    if (path_open) end_path ();
    path_open = true;
    funcs->move_to (path_start_x, path_start_y, user_data);
  }

  hb_position_t path_start_x;
  hb_position_t path_start_y;

  hb_position_t current_x;
  hb_position_t current_y;

  bool path_open;
  const hb_draw_funcs_t *funcs;
  void *user_data;
};
#endif

#endif /* HB_DRAW_HH */
