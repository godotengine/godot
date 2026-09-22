/**************************************************************************/
/*  editor_theme.cpp                                                      */
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

#include "editor_theme.h"

#include "editor/editor_string_names.h"
#include "scene/theme/theme_db.h"

Vector<StringName> EditorTheme::editor_theme_types;

// TODO: Refactor these and corresponding Theme methods to use the bool get_xxx(r_value) pattern internally.

// Keep in sync with Theme::get_color.
Color EditorTheme::get_color(const StringName &p_name, const StringName &p_theme_type) const {
	if (color_map.has(p_theme_type) && color_map[p_theme_type].has(p_name)) {
		return color_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme color '%s' in '%s'.", p_name, p_theme_type));
		}
		return Color();
	}
}

// Keep in sync with Theme::get_constant.
int EditorTheme::get_constant(const StringName &p_name, const StringName &p_theme_type) const {
	if (constant_map.has(p_theme_type) && constant_map[p_theme_type].has(p_name)) {
		return constant_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme constant '%s' in '%s'.", p_name, p_theme_type));
		}
		return 0;
	}
}

// Keep in sync with Theme::get_font.
Ref<Font> EditorTheme::get_font(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_map.has(p_theme_type) && font_map[p_theme_type].has(p_name) && font_map[p_theme_type][p_name].is_valid()) {
		return font_map[p_theme_type][p_name];
	} else if (has_default_font()) {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font '%s' in '%s'.", p_name, p_theme_type));
		}
		return default_font;
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_font();
	}
}

// Keep in sync with Theme::get_font_size.
int EditorTheme::get_font_size(const StringName &p_name, const StringName &p_theme_type) const {
	if (font_size_map.has(p_theme_type) && font_size_map[p_theme_type].has(p_name) && (font_size_map[p_theme_type][p_name] > 0)) {
		return font_size_map[p_theme_type][p_name];
	} else if (has_default_font_size()) {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font size '%s' in '%s'.", p_name, p_theme_type));
		}
		return default_font_size;
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme font size '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_font_size();
	}
}

// Keep in sync with Theme::get_icon.
Ref<Texture2D> EditorTheme::get_icon(const StringName &p_name, const StringName &p_theme_type) const {
	if (icon_map.has(p_theme_type) && icon_map[p_theme_type].has(p_name) && icon_map[p_theme_type][p_name].is_valid()) {
		return icon_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme icon '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_icon();
	}
}

// Keep in sync with Theme::get_stylebox.
Ref<StyleBox> EditorTheme::get_stylebox(const StringName &p_name, const StringName &p_theme_type) const {
	if (style_map.has(p_theme_type) && style_map[p_theme_type].has(p_name) && style_map[p_theme_type][p_name].is_valid()) {
		return style_map[p_theme_type][p_name];
	} else {
		if (editor_theme_types.has(p_theme_type)) {
			WARN_PRINT(vformat("Trying to access a non-existing editor theme stylebox '%s' in '%s'.", p_name, p_theme_type));
		}
		return ThemeDB::get_singleton()->get_fallback_stylebox();
	}
}

void EditorTheme::initialize() {
	editor_theme_types.append(EditorStringName(Editor));
	editor_theme_types.append(EditorStringName(EditorFonts));
	editor_theme_types.append(EditorStringName(EditorIcons));
	editor_theme_types.append(EditorStringName(EditorStyles));
}

void EditorTheme::finalize() {
	editor_theme_types.clear();
}
