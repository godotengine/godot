/*************************************************************************/
/*  godot_rect2.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "godot_rect2.h"

#include "math/math_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rect2_api_anchor() {
}

void GDAPI godot_rect2_new(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	*rect = Rect2();
}

void GDAPI godot_rect2_new_with_pos_and_size(godot_rect2 *p_rect, const godot_vector2 *p_pos, const godot_vector2 *p_size) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *pos = (const Vector2 *)p_pos;
	const Vector2 *size = (const Vector2 *)p_size;
	*rect = Rect2(*pos, *size);
}

godot_vector2 GDAPI *godot_rect2_get_pos(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	return (godot_vector2 *)&rect->pos;
}

void GDAPI godot_rect2_set_pos(godot_rect2 *p_rect, const godot_vector2 *p_pos) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *pos = (const Vector2 *)p_pos;
	rect->pos = *pos;
}

godot_vector2 GDAPI *godot_rect2_get_size(godot_rect2 *p_rect) {
	Rect2 *rect = (Rect2 *)p_rect;
	return (godot_vector2 *)&rect->size;
}

void GDAPI godot_rect2_set_size(godot_rect2 *p_rect, const godot_vector2 *p_size) {
	Rect2 *rect = (Rect2 *)p_rect;
	const Vector2 *size = (const Vector2 *)p_size;
	rect->size = *size;
}

#ifdef __cplusplus
}
#endif
