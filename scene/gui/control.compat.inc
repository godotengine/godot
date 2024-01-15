/**************************************************************************/
/*  control.compat.inc                                                    */
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

void Control::_add_theme_font_size_override_compat_87243(const StringName &p_name, int p_font_size) {
	add_theme_font_size_override(p_name, (float)p_font_size);
}

int Control::_get_theme_font_size_compat_87243(const StringName &p_name, const StringName &p_theme_type) const {
	return (int)get_theme_font_size(p_name, p_theme_type);
}

int Control::_get_theme_default_font_size_compat_87243() const {
	return (int)get_theme_default_font_size();
}

void Control::_add_theme_constant_override_compat_87243(const StringName &p_name, int p_constant) {
	add_theme_constant_override(p_name, (float)p_constant);
}

int Control::_get_theme_constant_compat_87243(const StringName &p_name, const StringName &p_theme_type) const {
	return (int)get_theme_constant(p_name, p_theme_type);
}

void Control::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("add_theme_font_size_override", "name", "font_size"), &Control::_add_theme_font_size_override_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_font_size", "name", "theme_type"), &Control::_get_theme_font_size_compat_87243, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_default_font_size"), &Control::_get_theme_default_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("add_theme_constant_override", "name", "font_size"), &Control::_add_theme_constant_override_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_constant", "name", "theme_type"), &Control::_get_theme_constant_compat_87243, DEFVAL(""));

	ClassDB::bind_compatibility_method(D_METHOD("get_theme_icon", "name", "theme_type"), &Control::get_theme_icon, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_stylebox", "name", "theme_type"), &Control::get_theme_stylebox, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_font", "name", "theme_type"), &Control::get_theme_font, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_font_size", "name", "theme_type"), &Control::get_theme_font_size, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_color", "name", "theme_type"), &Control::get_theme_color, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("get_theme_constant", "name", "theme_type"), &Control::get_theme_constant, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_icon", "name", "theme_type"), &Control::has_theme_icon, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_stylebox", "name", "theme_type"), &Control::has_theme_stylebox, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_font", "name", "theme_type"), &Control::has_theme_font, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_font_size", "name", "theme_type"), &Control::has_theme_font_size, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_color", "name", "theme_type"), &Control::has_theme_color, DEFVAL(""));
	ClassDB::bind_compatibility_method(D_METHOD("has_theme_constant", "name", "theme_type"), &Control::has_theme_constant, DEFVAL(""));
}

#endif
