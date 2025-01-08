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

#include "hb-draw.hh"

/**
 * SECTION:hb-draw
 * @title: hb-draw
 * @short_description: Glyph drawing
 * @include: hb.h
 *
 * Functions for drawing (extracting) glyph shapes.
 *
 * The #hb_draw_funcs_t struct can be used with hb_font_draw_glyph().
 **/

static void
hb_draw_move_to_nil (hb_draw_funcs_t *dfuncs HB_UNUSED, void *draw_data HB_UNUSED,
		     hb_draw_state_t *st HB_UNUSED,
		     float to_x HB_UNUSED, float to_y HB_UNUSED,
		     void *user_data HB_UNUSED) {}

static void
hb_draw_line_to_nil (hb_draw_funcs_t *dfuncs HB_UNUSED, void *draw_data HB_UNUSED,
		     hb_draw_state_t *st HB_UNUSED,
		     float to_x HB_UNUSED, float to_y HB_UNUSED,
		     void *user_data HB_UNUSED) {}

static void
hb_draw_quadratic_to_nil (hb_draw_funcs_t *dfuncs, void *draw_data,
			  hb_draw_state_t *st,
			  float control_x, float control_y,
			  float to_x, float to_y,
			  void *user_data HB_UNUSED)
{
#define HB_ONE_THIRD 0.33333333f
  dfuncs->emit_cubic_to (draw_data, *st,
			 (st->current_x + 2.f * control_x) * HB_ONE_THIRD,
			 (st->current_y + 2.f * control_y) * HB_ONE_THIRD,
			 (to_x + 2.f * control_x) * HB_ONE_THIRD,
			 (to_y + 2.f * control_y) * HB_ONE_THIRD,
			 to_x, to_y);
#undef HB_ONE_THIRD
}

static void
hb_draw_cubic_to_nil (hb_draw_funcs_t *dfuncs HB_UNUSED, void *draw_data HB_UNUSED,
		      hb_draw_state_t *st HB_UNUSED,
		      float control1_x HB_UNUSED, float control1_y HB_UNUSED,
		      float control2_x HB_UNUSED, float control2_y HB_UNUSED,
		      float to_x HB_UNUSED, float to_y HB_UNUSED,
		      void *user_data HB_UNUSED) {}

static void
hb_draw_close_path_nil (hb_draw_funcs_t *dfuncs HB_UNUSED, void *draw_data HB_UNUSED,
			hb_draw_state_t *st HB_UNUSED,
			void *user_data HB_UNUSED) {}


static bool
_hb_draw_funcs_set_preamble (hb_draw_funcs_t    *dfuncs,
			     bool                func_is_null,
			     void              **user_data,
			     hb_destroy_func_t  *destroy)
{
  if (hb_object_is_immutable (dfuncs))
  {
    if (*destroy)
      (*destroy) (*user_data);
    return false;
  }

  if (func_is_null)
  {
    if (*destroy)
      (*destroy) (*user_data);
    *destroy = nullptr;
    *user_data = nullptr;
  }

  return true;
}

static bool
_hb_draw_funcs_set_middle (hb_draw_funcs_t   *dfuncs,
			   void              *user_data,
			   hb_destroy_func_t  destroy)
{
  if (user_data && !dfuncs->user_data)
  {
    dfuncs->user_data = (decltype (dfuncs->user_data)) hb_calloc (1, sizeof (*dfuncs->user_data));
    if (unlikely (!dfuncs->user_data))
      goto fail;
  }
  if (destroy && !dfuncs->destroy)
  {
    dfuncs->destroy = (decltype (dfuncs->destroy)) hb_calloc (1, sizeof (*dfuncs->destroy));
    if (unlikely (!dfuncs->destroy))
      goto fail;
  }

  return true;

fail:
  if (destroy)
    (destroy) (user_data);
  return false;
}

#define HB_DRAW_FUNC_IMPLEMENT(name)						\
										\
void										\
hb_draw_funcs_set_##name##_func (hb_draw_funcs_t	 *dfuncs,		\
				 hb_draw_##name##_func_t  func,			\
				 void			 *user_data,		\
				 hb_destroy_func_t	  destroy)		\
{										\
  if (!_hb_draw_funcs_set_preamble (dfuncs, !func, &user_data, &destroy))\
      return;                                                            \
										\
  if (dfuncs->destroy && dfuncs->destroy->name)					\
    dfuncs->destroy->name (!dfuncs->user_data ? nullptr : dfuncs->user_data->name); \
									 \
  if (!_hb_draw_funcs_set_middle (dfuncs, user_data, destroy))           \
      return;                                                            \
									\
  if (func)								\
    dfuncs->func.name = func;						\
  else									\
    dfuncs->func.name = hb_draw_##name##_nil;				\
									\
  if (dfuncs->user_data)						\
    dfuncs->user_data->name = user_data;				\
  if (dfuncs->destroy)							\
    dfuncs->destroy->name = destroy;					\
}

HB_DRAW_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_DRAW_FUNC_IMPLEMENT

/**
 * hb_draw_funcs_create:
 *
 * Creates a new draw callbacks object.
 *
 * Return value: (transfer full):
 * A newly allocated #hb_draw_funcs_t with a reference count of 1. The initial
 * reference count should be released with hb_draw_funcs_destroy when you are
 * done using the #hb_draw_funcs_t. This function never returns `NULL`. If
 * memory cannot be allocated, a special singleton #hb_draw_funcs_t object will
 * be returned.
 *
 * Since: 4.0.0
 **/
hb_draw_funcs_t *
hb_draw_funcs_create ()
{
  hb_draw_funcs_t *dfuncs;
  if (unlikely (!(dfuncs = hb_object_create<hb_draw_funcs_t> ())))
    return const_cast<hb_draw_funcs_t *> (&Null (hb_draw_funcs_t));

  dfuncs->func =  Null (hb_draw_funcs_t).func;

  return dfuncs;
}

DEFINE_NULL_INSTANCE (hb_draw_funcs_t) =
{
  HB_OBJECT_HEADER_STATIC,

  {
#define HB_DRAW_FUNC_IMPLEMENT(name) hb_draw_##name##_nil,
    HB_DRAW_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_DRAW_FUNC_IMPLEMENT
  }
};

/**
 * hb_draw_funcs_get_empty:
 *
 * Fetches the singleton empty draw-functions structure.
 *
 * Return value: (transfer full): The empty draw-functions structure
 *
 * Since: 7.0.0
 **/
hb_draw_funcs_t *
hb_draw_funcs_get_empty ()
{
  return const_cast<hb_draw_funcs_t *> (&Null (hb_draw_funcs_t));
}

/**
 * hb_draw_funcs_reference: (skip)
 * @dfuncs: draw functions
 *
 * Increases the reference count on @dfuncs by one.
 *
 * This prevents @dfuncs from being destroyed until a matching
 * call to hb_draw_funcs_destroy() is made.
 *
 * Return value: (transfer full):
 * The referenced #hb_draw_funcs_t.
 *
 * Since: 4.0.0
 **/
hb_draw_funcs_t *
hb_draw_funcs_reference (hb_draw_funcs_t *dfuncs)
{
  return hb_object_reference (dfuncs);
}

/**
 * hb_draw_funcs_destroy: (skip)
 * @dfuncs: draw functions
 *
 * Deallocate the @dfuncs.
 * Decreases the reference count on @dfuncs by one. If the result is zero, then
 * @dfuncs and all associated resources are freed. See hb_draw_funcs_reference().
 *
 * Since: 4.0.0
 **/
void
hb_draw_funcs_destroy (hb_draw_funcs_t *dfuncs)
{
  if (!hb_object_destroy (dfuncs)) return;

  if (dfuncs->destroy)
  {
#define HB_DRAW_FUNC_IMPLEMENT(name) \
    if (dfuncs->destroy->name) dfuncs->destroy->name (!dfuncs->user_data ? nullptr : dfuncs->user_data->name);
      HB_DRAW_FUNCS_IMPLEMENT_CALLBACKS
#undef HB_DRAW_FUNC_IMPLEMENT
  }

  hb_free (dfuncs->destroy);
  hb_free (dfuncs->user_data);

  hb_free (dfuncs);
}

/**
 * hb_draw_funcs_set_user_data: (skip)
 * @dfuncs: The draw-functions structure
 * @key: The user-data key
 * @data: A pointer to the user data
 * @destroy: (nullable): A callback to call when @data is not needed anymore
 * @replace: Whether to replace an existing data with the same key
 *
 * Attaches a user-data key/data pair to the specified draw-functions structure. 
 *
 * Return value: `true` if success, `false` otherwise
 *
 * Since: 7.0.0
 **/
hb_bool_t
hb_draw_funcs_set_user_data (hb_draw_funcs_t *dfuncs,
			     hb_user_data_key_t *key,
			     void *              data,
			     hb_destroy_func_t   destroy,
			     hb_bool_t           replace)
{
  return hb_object_set_user_data (dfuncs, key, data, destroy, replace);
}

/**
 * hb_draw_funcs_get_user_data: (skip)
 * @dfuncs: The draw-functions structure
 * @key: The user-data key to query
 *
 * Fetches the user-data associated with the specified key,
 * attached to the specified draw-functions structure.
 *
 * Return value: (transfer none): A pointer to the user data
 *
 * Since: 7.0.0
 **/
void *
hb_draw_funcs_get_user_data (const hb_draw_funcs_t *dfuncs,
			     hb_user_data_key_t       *key)
{
  return hb_object_get_user_data (dfuncs, key);
}

/**
 * hb_draw_funcs_make_immutable:
 * @dfuncs: draw functions
 *
 * Makes @dfuncs object immutable.
 *
 * Since: 4.0.0
 **/
void
hb_draw_funcs_make_immutable (hb_draw_funcs_t *dfuncs)
{
  if (hb_object_is_immutable (dfuncs))
    return;

  hb_object_make_immutable (dfuncs);
}

/**
 * hb_draw_funcs_is_immutable:
 * @dfuncs: draw functions
 *
 * Checks whether @dfuncs is immutable.
 *
 * Return value: `true` if @dfuncs is immutable, `false` otherwise
 *
 * Since: 4.0.0
 **/
hb_bool_t
hb_draw_funcs_is_immutable (hb_draw_funcs_t *dfuncs)
{
  return hb_object_is_immutable (dfuncs);
}


/**
 * hb_draw_move_to:
 * @dfuncs: draw functions
 * @draw_data: associated draw data passed by the caller
 * @st: current draw state
 * @to_x: X component of target point
 * @to_y: Y component of target point
 *
 * Perform a "move-to" draw operation.
 *
 * Since: 4.0.0
 **/
void
hb_draw_move_to (hb_draw_funcs_t *dfuncs, void *draw_data,
		 hb_draw_state_t *st,
		 float to_x, float to_y)
{
  dfuncs->move_to (draw_data, *st,
		   to_x, to_y);
}

/**
 * hb_draw_line_to:
 * @dfuncs: draw functions
 * @draw_data: associated draw data passed by the caller
 * @st: current draw state
 * @to_x: X component of target point
 * @to_y: Y component of target point
 *
 * Perform a "line-to" draw operation.
 *
 * Since: 4.0.0
 **/
void
hb_draw_line_to (hb_draw_funcs_t *dfuncs, void *draw_data,
		 hb_draw_state_t *st,
		 float to_x, float to_y)
{
  dfuncs->line_to (draw_data, *st,
		   to_x, to_y);
}

/**
 * hb_draw_quadratic_to:
 * @dfuncs: draw functions
 * @draw_data: associated draw data passed by the caller
 * @st: current draw state
 * @control_x: X component of control point
 * @control_y: Y component of control point
 * @to_x: X component of target point
 * @to_y: Y component of target point
 *
 * Perform a "quadratic-to" draw operation.
 *
 * Since: 4.0.0
 **/
void
hb_draw_quadratic_to (hb_draw_funcs_t *dfuncs, void *draw_data,
		      hb_draw_state_t *st,
		      float control_x, float control_y,
		      float to_x, float to_y)
{
  dfuncs->quadratic_to (draw_data, *st,
			control_x, control_y,
			to_x, to_y);
}

/**
 * hb_draw_cubic_to:
 * @dfuncs: draw functions
 * @draw_data: associated draw data passed by the caller
 * @st: current draw state
 * @control1_x: X component of first control point
 * @control1_y: Y component of first control point
 * @control2_x: X component of second control point
 * @control2_y: Y component of second control point
 * @to_x: X component of target point
 * @to_y: Y component of target point
 *
 * Perform a "cubic-to" draw operation.
 *
 * Since: 4.0.0
 **/
void
hb_draw_cubic_to (hb_draw_funcs_t *dfuncs, void *draw_data,
		  hb_draw_state_t *st,
		  float control1_x, float control1_y,
		  float control2_x, float control2_y,
		  float to_x, float to_y)
{
  dfuncs->cubic_to (draw_data, *st,
		    control1_x, control1_y,
		    control2_x, control2_y,
		    to_x, to_y);
}

/**
 * hb_draw_close_path:
 * @dfuncs: draw functions
 * @draw_data: associated draw data passed by the caller
 * @st: current draw state
 *
 * Perform a "close-path" draw operation.
 *
 * Since: 4.0.0
 **/
void
hb_draw_close_path (hb_draw_funcs_t *dfuncs, void *draw_data,
		    hb_draw_state_t *st)
{
  dfuncs->close_path (draw_data, *st);
}


#endif
