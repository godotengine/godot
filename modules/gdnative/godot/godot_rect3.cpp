/*************************************************************************/
/*  godot_rect3.cpp                                                      */
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
#include "godot_rect3.h"

#include "math/rect3.h"

#ifdef __cplusplus
extern "C" {
#endif

void _rect3_api_anchor() {
}

void GDAPI godot_rect3_new(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	*rect = Rect3();
}

void GDAPI godot_rect3_new_with_pos_and_size(godot_rect3 *p_rect, const godot_vector3 *p_pos, const godot_vector3 *p_size) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *pos = (const Vector3 *)p_pos;
	const Vector3 *size = (const Vector3 *)p_size;
	*rect = Rect3(*pos, *size);
}

godot_vector3 GDAPI *godot_rect3_get_pos(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	return (godot_vector3 *)&rect->pos;
}

void GDAPI godot_rect3_set_pos(godot_rect3 *p_rect, const godot_vector3 *p_pos) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *pos = (const Vector3 *)p_pos;
	rect->pos = *pos;
}

godot_vector3 GDAPI *godot_rect3_get_size(godot_rect3 *p_rect) {
	Rect3 *rect = (Rect3 *)p_rect;
	return (godot_vector3 *)&rect->size;
}

void GDAPI godot_rect3_set_size(godot_rect3 *p_rect, const godot_vector3 *p_size) {
	Rect3 *rect = (Rect3 *)p_rect;
	const Vector3 *size = (const Vector3 *)p_size;
	rect->size = *size;
}

#ifdef __cplusplus
}
#endif
