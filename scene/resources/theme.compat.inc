/**************************************************************************/
/*  theme.compat.inc                                                      */
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

void Theme::_set_default_font_size_compat_87243(int p_font_size) {
	set_default_font_size((float)p_font_size);
}

int Theme::_get_default_font_size_compat_87243() const {
	return (int)get_default_font_size();
}

void Theme::_set_font_size_compat_87243(const StringName &p_name, const StringName &p_theme_type, int p_font_size) {
	set_font_size(p_name, p_theme_type, (float)p_font_size);
}

int Theme::_get_font_size_compat_87243(const StringName &p_name, const StringName &p_theme_type) const {
	return (int)get_font_size(p_name, p_theme_type);
}

void Theme::_set_constant_compat_87243(const StringName &p_name, const StringName &p_theme_type, int p_constant) {
	set_constant(p_name, p_theme_type, (float)p_constant);
}

int Theme::_get_constant_compat_87243(const StringName &p_name, const StringName &p_theme_type) const {
	return (int)get_constant(p_name, p_theme_type);
}

void Theme::_bind_compatibility_methods() {
	ClassDB::bind_compatibility_method(D_METHOD("set_font_size", "name", "theme_type", "font_size"), &Theme::_set_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("get_font_size", "name", "theme_type"), &Theme::_get_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("set_default_font_size", "font_size"), &Theme::_set_default_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("get_default_font_size"), &Theme::_get_default_font_size_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("set_constant", "name", "theme_type", "constant"), &Theme::_set_constant_compat_87243);
	ClassDB::bind_compatibility_method(D_METHOD("get_constant", "name", "theme_type"), &Theme::_get_constant_compat_87243);
}

#endif
