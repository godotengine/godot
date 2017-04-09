/*************************************************************************/
/*  godot_color.cpp                                                      */
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
#include "godot_color.h"

#include "color.h"

#ifdef __cplusplus
extern "C" {
#endif

void _color_api_anchor() {
}

void GDAPI godot_color_new(godot_color *p_color) {
	Color *color = (Color *)p_color;
	*color = Color();
}

void GDAPI godot_color_new_rgba(godot_color *p_color, const godot_real r, const godot_real g, const godot_real b, const godot_real a) {
	Color *color = (Color *)p_color;
	*color = Color(r, g, b, a);
}

uint32_t GDAPI godot_color_get_32(const godot_color *p_color) {
	const Color *color = (const Color *)p_color;
	return color->to_32();
}

float GDAPI *godot_color_index(godot_color *p_color, const godot_int idx) {
	Color *color = (Color *)p_color;
	return &color->operator[](idx);
}

#ifdef __cplusplus
}
#endif
