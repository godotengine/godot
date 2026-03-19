/*
 * Copyright © 2022 Behdad Esfahbod
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

#ifndef HB_NO_PAINT

#include "hb-paint-bounded.hh"

#include "hb-machinery.hh"


/*
 * This file implements boundedness computation of COLRv1 fonts as described in:
 *
 * https://learn.microsoft.com/en-us/typography/opentype/spec/colr#glyph-metrics-and-boundedness
 */

static void
hb_paint_bounded_push_clip_glyph (hb_paint_funcs_t *funcs HB_UNUSED,
				  void *paint_data,
				  hb_codepoint_t glyph,
				  hb_font_t *font,
				  void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->push_clip ();
}

static void
hb_paint_bounded_push_clip_rectangle (hb_paint_funcs_t *funcs HB_UNUSED,
				      void *paint_data,
				      float xmin, float ymin, float xmax, float ymax,
				      void *user_data)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->push_clip ();
}

static void
hb_paint_bounded_pop_clip (hb_paint_funcs_t *funcs HB_UNUSED,
			   void *paint_data,
			   void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->pop_clip ();
}

static void
hb_paint_bounded_push_group (hb_paint_funcs_t *funcs HB_UNUSED,
			     void *paint_data,
			     void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->push_group ();
}

static void
hb_paint_bounded_pop_group (hb_paint_funcs_t *funcs HB_UNUSED,
			    void *paint_data,
			    hb_paint_composite_mode_t mode,
			    void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->pop_group (mode);
}

static hb_bool_t
hb_paint_bounded_paint_image (hb_paint_funcs_t *funcs HB_UNUSED,
			      void *paint_data,
			      hb_blob_t *blob HB_UNUSED,
			      unsigned int width HB_UNUSED,
			      unsigned int height HB_UNUSED,
			      hb_tag_t format HB_UNUSED,
			      float slant HB_UNUSED,
			      hb_glyph_extents_t *glyph_extents,
			      void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->push_clip ();
  c->paint ();
  c->pop_clip ();

  return true;
}

static void
hb_paint_bounded_paint_color (hb_paint_funcs_t *funcs HB_UNUSED,
			      void *paint_data,
			      hb_bool_t use_foreground HB_UNUSED,
			      hb_color_t color HB_UNUSED,
			      void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->paint ();
}

static void
hb_paint_bounded_paint_linear_gradient (hb_paint_funcs_t *funcs HB_UNUSED,
				        void *paint_data,
				        hb_color_line_t *color_line HB_UNUSED,
				        float x0 HB_UNUSED, float y0 HB_UNUSED,
				        float x1 HB_UNUSED, float y1 HB_UNUSED,
				        float x2 HB_UNUSED, float y2 HB_UNUSED,
				        void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->paint ();
}

static void
hb_paint_bounded_paint_radial_gradient (hb_paint_funcs_t *funcs HB_UNUSED,
				        void *paint_data,
				        hb_color_line_t *color_line HB_UNUSED,
				        float x0 HB_UNUSED, float y0 HB_UNUSED, float r0 HB_UNUSED,
				        float x1 HB_UNUSED, float y1 HB_UNUSED, float r1 HB_UNUSED,
				        void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->paint ();
}

static void
hb_paint_bounded_paint_sweep_gradient (hb_paint_funcs_t *funcs HB_UNUSED,
				       void *paint_data,
				       hb_color_line_t *color_line HB_UNUSED,
				       float cx HB_UNUSED, float cy HB_UNUSED,
				       float start_angle HB_UNUSED,
				       float end_angle HB_UNUSED,
				       void *user_data HB_UNUSED)
{
  hb_paint_bounded_context_t *c = (hb_paint_bounded_context_t *) paint_data;

  c->paint ();
}

static inline void free_static_paint_bounded_funcs ();

static struct hb_paint_bounded_funcs_lazy_loader_t : hb_paint_funcs_lazy_loader_t<hb_paint_bounded_funcs_lazy_loader_t>
{
  static hb_paint_funcs_t *create ()
  {
    hb_paint_funcs_t *funcs = hb_paint_funcs_create ();

    hb_paint_funcs_set_push_clip_glyph_func (funcs, hb_paint_bounded_push_clip_glyph, nullptr, nullptr);
    hb_paint_funcs_set_push_clip_rectangle_func (funcs, hb_paint_bounded_push_clip_rectangle, nullptr, nullptr);
    hb_paint_funcs_set_pop_clip_func (funcs, hb_paint_bounded_pop_clip, nullptr, nullptr);
    hb_paint_funcs_set_push_group_func (funcs, hb_paint_bounded_push_group, nullptr, nullptr);
    hb_paint_funcs_set_pop_group_func (funcs, hb_paint_bounded_pop_group, nullptr, nullptr);
    hb_paint_funcs_set_color_func (funcs, hb_paint_bounded_paint_color, nullptr, nullptr);
    hb_paint_funcs_set_image_func (funcs, hb_paint_bounded_paint_image, nullptr, nullptr);
    hb_paint_funcs_set_linear_gradient_func (funcs, hb_paint_bounded_paint_linear_gradient, nullptr, nullptr);
    hb_paint_funcs_set_radial_gradient_func (funcs, hb_paint_bounded_paint_radial_gradient, nullptr, nullptr);
    hb_paint_funcs_set_sweep_gradient_func (funcs, hb_paint_bounded_paint_sweep_gradient, nullptr, nullptr);

    hb_paint_funcs_make_immutable (funcs);

    hb_atexit (free_static_paint_bounded_funcs);

    return funcs;
  }
} static_paint_bounded_funcs;

static inline
void free_static_paint_bounded_funcs ()
{
  static_paint_bounded_funcs.free_instance ();
}

hb_paint_funcs_t *
hb_paint_bounded_get_funcs ()
{
  return static_paint_bounded_funcs.get_unconst ();
}


#endif
