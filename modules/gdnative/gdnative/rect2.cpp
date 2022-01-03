/*************************************************************************/
/*  rect2.cpp                                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gdnative/rect2.h"

#include "core/math/rect2.h"

static_assert(sizeof(godot_rect2) == sizeof(Rect2), "Rect2 size mismatch");
static_assert(sizeof(godot_rect2i) == sizeof(Rect2i), "Rect2i size mismatch");

#ifdef __cplusplus
extern "C" {
#endif

void GDAPI godot_rect2_new(godot_rect2 *p_self) {
	memnew_placement(p_self, Rect2);
}

void GDAPI godot_rect2_new_copy(godot_rect2 *r_dest, const godot_rect2 *p_src) {
	memnew_placement(r_dest, Rect2(*(Rect2 *)p_src));
}

void GDAPI godot_rect2i_new(godot_rect2i *p_self) {
	memnew_placement(p_self, Rect2i);
}

void GDAPI godot_rect2i_new_copy(godot_rect2i *r_dest, const godot_rect2i *p_src) {
	memnew_placement(r_dest, Rect2i(*(Rect2i *)p_src));
}

#ifdef __cplusplus
}
#endif
