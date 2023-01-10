/**************************************************************************/
/*  color.h                                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GDNATIVE_COLOR_H
#define GDNATIVE_COLOR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define GODOT_COLOR_SIZE 16

#ifndef GODOT_CORE_API_GODOT_COLOR_TYPE_DEFINED
#define GODOT_CORE_API_GODOT_COLOR_TYPE_DEFINED
typedef struct {
	uint8_t _dont_touch_that[GODOT_COLOR_SIZE];
} godot_color;
#endif

// reduce extern "C" nesting for VS2013
#ifdef __cplusplus
}
#endif

#include <gdnative/gdnative.h>
#include <gdnative/string.h>

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_color_new_rgba(godot_color *r_dest, const godot_real p_r, const godot_real p_g, const godot_real p_b, const godot_real p_a);
void GDAPI godot_color_new_rgb(godot_color *r_dest, const godot_real p_r, const godot_real p_g, const godot_real p_b);

godot_real godot_color_get_r(const godot_color *p_self);
void godot_color_set_r(godot_color *p_self, const godot_real r);

godot_real godot_color_get_g(const godot_color *p_self);
void godot_color_set_g(godot_color *p_self, const godot_real g);

godot_real godot_color_get_b(const godot_color *p_self);
void godot_color_set_b(godot_color *p_self, const godot_real b);

godot_real godot_color_get_a(const godot_color *p_self);
void godot_color_set_a(godot_color *p_self, const godot_real a);

godot_real godot_color_get_h(const godot_color *p_self);
godot_real godot_color_get_s(const godot_color *p_self);
godot_real godot_color_get_v(const godot_color *p_self);

godot_string GDAPI godot_color_as_string(const godot_color *p_self);

godot_int GDAPI godot_color_to_rgba32(const godot_color *p_self);

godot_int GDAPI godot_color_to_abgr32(const godot_color *p_self);

godot_int GDAPI godot_color_to_abgr64(const godot_color *p_self);

godot_int GDAPI godot_color_to_argb64(const godot_color *p_self);

godot_int GDAPI godot_color_to_rgba64(const godot_color *p_self);

godot_int GDAPI godot_color_to_argb32(const godot_color *p_self);

godot_real GDAPI godot_color_gray(const godot_color *p_self);

godot_color GDAPI godot_color_inverted(const godot_color *p_self);

godot_color GDAPI godot_color_contrasted(const godot_color *p_self);

godot_color GDAPI godot_color_linear_interpolate(const godot_color *p_self, const godot_color *p_b, const godot_real p_t);

godot_color GDAPI godot_color_blend(const godot_color *p_self, const godot_color *p_over);

godot_color GDAPI godot_color_darkened(const godot_color *p_self, const godot_real p_amount);

godot_color GDAPI godot_color_from_hsv(const godot_color *p_self, const godot_real p_h, const godot_real p_s, const godot_real p_v, const godot_real p_a);

godot_color GDAPI godot_color_lightened(const godot_color *p_self, const godot_real p_amount);

godot_string GDAPI godot_color_to_html(const godot_color *p_self, const godot_bool p_with_alpha);

godot_bool GDAPI godot_color_operator_equal(const godot_color *p_self, const godot_color *p_b);

godot_bool GDAPI godot_color_operator_less(const godot_color *p_self, const godot_color *p_b);

#ifdef __cplusplus
}
#endif

#endif // GDNATIVE_COLOR_H
