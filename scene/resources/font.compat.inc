/**************************************************************************/
/*  font.compat.inc                                                       */
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

#ifndef DISABLE_DEPRECATED

RID Font::_find_variation_compat_80954(const Dictionary &p_variation_coordinates, int p_face_index, float p_strength, Transform2D p_transform) const {
	return find_variation(p_variation_coordinates, p_face_index, p_strength, p_transform, 0, 0, 0, 0, 0.0);
}

RID Font::_find_variation_compat_87668(const Dictionary &p_variation_coordinates, int p_face_index, float p_strength, Transform2D p_transform, int p_spacing_top, int p_spacing_bottom, int p_spacing_space, int p_spacing_glyph) const {
	return find_variation(p_variation_coordinates, p_face_index, p_strength, p_transform, p_spacing_top, p_spacing_bottom, p_spacing_space, p_spacing_glyph, 0.0);
}

void Font::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("find_variation", "variation_coordinates", "face_index", "strength", "transform"), &Font::_find_variation_compat_80954, DEFVAL(0), DEFVAL(0.0), DEFVAL(Transform2D()));
	ClassDB::bind_compatibility_method(D_METHOD("find_variation", "variation_coordinates", "face_index", "strength", "transform", "spacing_top", "spacing_bottom", "spacing_space", "spacing_glyph"), &Font::_find_variation_compat_87668, DEFVAL(0), DEFVAL(0.0), DEFVAL(Transform2D()), DEFVAL(0), DEFVAL(0), DEFVAL(0), DEFVAL(0));
}

#endif
